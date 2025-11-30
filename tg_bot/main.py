from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.filters import CommandStart
from dotenv import load_dotenv
import os
import sys
import asyncio
import json
import logging

from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True)]
)
logger = logging.getLogger("bchef.bot")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

from rl.bandit import BanditPolicy

bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()

# Load recipes
with open("recipes.json", "r", encoding="utf-8") as f:
    recipes = json.load(f)

RECIPE_LIST = {
    "🍳 Fried Eggs": "fried eggs",
    "🍝 Pasta": "pasta",
    "🐟 Fried Fish": "fried fish",
    "🥩 Fried Meat": "fried meat",
    "🍟 Fried Potatoes": "fried potatoes"
}

user_data = {}

# rl police init
bandit_policy = BanditPolicy(path="qtable.json")

# tones dir
TONES_DIR = os.path.join(BASE_DIR, "rl", "tones")
_template_cache = {}


def _load_tone_templates(tone: str):
    if tone in _template_cache:
        return _template_cache[tone]

    path = os.path.join(TONES_DIR, f"{tone}.txt")
    logger.info(f"Loading tone template: {path}")
    if not os.path.exists(path):
        path = os.path.join(TONES_DIR, "neutral.txt")
        logger.warning(f"Tone file not found: {path}, falling back to neutral")
        if not os.path.exists(path):
            logger.error(f"Template file not found for tone: {tone}")
            return None

    templates = {"correct": "", "incorrect": "", "correct_last": "", "incorrect_last": ""}
    current_key = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current_key = line[1:-1]
                elif current_key and line:
                    templates[current_key] += line + "\n"
    except FileNotFoundError as e:
        logger.error(f"Error loading template {tone}: {e}")
        # Hardcoded fallback if files are truly missing
        return {
            "correct": "Great job with the {food}! Looking good.",
            "incorrect": "Hmm, this doesn't look quite right. You should be {recipe_step}.",
            "correct_last": "Perfect! Your {food} is ready to serve.",
            "incorrect_last": "This {food} needs more work. Let's try again."
        }

    for key in templates:
        templates[key] = templates[key].strip()

    _template_cache[tone] = templates
    return templates


def _get_template(tone: str, is_correct: bool, is_last: bool) -> str:
    templates = _load_tone_templates(tone)
    logger.warning(f"No templates found for tone: {tone}")
    if not templates:
        return None

    key = ("correct" if is_correct else "incorrect") + ("_last" if is_last else "")
    template = templates.get(key)

    if not template:
        fallback_key = "correct" if is_correct else "incorrect"
        template = templates.get(fallback_key, "")
        logger.warning(f"Template {key} not found, using {fallback_key}")

    return template

# --- Keyboards ---

def mode_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🍳 Gordon Ramsay"), KeyboardButton(text="👵 Sweet Grandma"),
             KeyboardButton(text="👨‍🍳 None of them")]
        ],
        resize_keyboard=True
    )

def recipes_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🍳 Fried Eggs"), KeyboardButton(text="🍝 Pasta")],
            [KeyboardButton(text="🐟 Fried Fish")],
            [KeyboardButton(text="🍟 Fried Potatoes"),KeyboardButton(text="🥩 Fried Meat")],
        ],
        resize_keyboard=True
    )

def actions_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📖 View Recipe")],
            [KeyboardButton(text="🍲 Cook a New Dish")],
            [KeyboardButton(text="✅ Finish Cooking")]
        ],
        resize_keyboard=True
    )

def feedback_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="❤️ Great!", callback_data="feedback_like"),
                InlineKeyboardButton(text="👎 Not so good", callback_data="feedback_dislike")
            ]
        ]
    )

def determine_state(result: dict, user_data: dict):
    recipe_key = user_data.get("recipe")
    current_step_idx = user_data.get("step", 0)

    if not recipe_key or recipe_key not in recipes:
        return 0

    recipe = recipes[recipe_key]
    expected_list = recipe.get("expected_per_step", [])

    if not expected_list or current_step_idx >= len(expected_list):
        return 0

    expected = expected_list[current_step_idx]

    food = "unknown"
    doneness = "unknown"
    container = "unknown"
    action = None

    if result.get("type") == "image":
        photo = result.get("photo", {})
        food = photo.get("food", "unknown")
        doneness = photo.get("doneness", "unknown")
        container = photo.get("container", "unknown")
    elif result.get("type") == "video":
        fusion = result.get("fusion", {})
        report = fusion.get("report", {})
        food = report.get("food") or report.get("photo_top1") or "unknown"
        doneness = report.get("doneness", "unknown")
        container = report.get("container", "unknown")
        action = report.get("action")

    food = str(food).lower()
    doneness = str(doneness).lower()
    container = str(container).lower()
    action = str(action).lower() if action else None

    if expected.get("food") and food in [f.lower() for f in expected["food"]]:
        return 1
    if expected.get("action") and action and action in [a.lower() for a in expected["action"]]:
        return 1
    if expected.get("doneness") and doneness in [d.lower() for d in expected["doneness"]]:
        return 1
    if expected.get("container") and container in [c.lower() for c in expected["container"]]:
        return 1
    return 0

async def format_inference_response(result: dict, user_id: int) -> str:

    if not isinstance(result, dict):
        return "I couldn't analyze your media. Please try again with a clearer photo or video."

    result_type = result.get("type")

    user_recipe = user_data.get(user_id, {}).get("recipe")
    user_step = user_data.get(user_id, {}).get("step", 0)
    steps = recipes.get(user_recipe, {}).get("steps", [])
    is_last = not user_data.get(user_id, {}).get("cooking", True) or (steps and user_step >= len(steps))

    current_step_text = ""
    if steps and user_step < len(steps):
        current_step_text = steps[user_step].lower()

    # === RL: check state and choose tone ===
    # state = determine_state(result, user_data.get(user_id, {}))
    state = determine_state(result, user_data[user_id])

    is_correct = state == 1
    tone = bandit_policy.choose_action(user_id, state, user_data)

    logger.info(f"RL Debug - User: {user_id}, State: {state}, Tone: {tone}, Correct: {is_correct}")

    # for q update after feedback
    user_data[user_id]["last_tone"] = tone
    user_data[user_id]["last_state"] = state
    user_data[user_id]["last_result"] = result
    # =========================================

    if result_type == "image":
        return await format_photo_response(result, is_last, tone, is_correct, current_step_text)
    elif result_type == "video":
        return await format_video_response(result, is_last, tone, is_correct, current_step_text)
    else:
        return "I received your media but couldn't process it properly. Please try again."
    

def _is_unknown(value: str) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    s = value.strip().lower()
    if s == "" or "unknown" in s or s == "unknown food type.":
        return True
    return False

def _format_analysis_lines(pairs):
    """
    pairs: list of tuples (emoji_and_label, value)
    returns formatted string with only known values (each on its own line)
    """
    lines = []
    for label, value in pairs:
        if not _is_unknown(value):
            lines.append(f"• {label}: {value}")
    if not lines:
        return ""
    return "\n\n🔍 My analysis:\n" + "\n".join(lines)

async def format_photo_response(result: dict, is_last: bool,
                                tone: str = "neutral", is_correct: bool = True,
                                recipe_step: str = "") -> str:
    # try fused report first, then photo dict (more robust)
    photo_data = result.get("photo", {}) or result.get("fusion", {}).get("report", {})

    food = photo_data.get("food") or photo_data.get("photo_top1")
    doneness = photo_data.get("doneness") or photo_data.get("photo_doneness")
    container = photo_data.get("container") or photo_data.get("photo_container")
    recommendation = photo_data.get("recommendation") or photo_data.get("photo_recommendation", "")

    template = _get_template(tone, is_correct, is_last)
    if template:
        try:
            base_msg = template.format(
                food=food or "unknown dish",
                doneness=doneness or "unknown doneness",
                container=container or "unknown container",
                recommendation=recommendation or "",
                recipe_step=recipe_step
            )
        except KeyError:
            logger.info("Error during template matching")
            base_msg = template
    else:
        if is_last:
            base_msg = f"🎉 Perfect! I see your {food or 'dish'} is ready!"
        else:
            base_msg = f"👨‍🍳 Great! I see you're working on {food or 'that dish'}."

    # build analysis only with known fields
    analysis = _format_analysis_lines([
        ("🍽️ Food", food),
        ("🔥 Doneness", doneness),
        ("🍳 Container", container),
    ])

    # recommendation only if meaningful
    if recommendation and not _is_unknown(recommendation):
        if analysis:
            analysis += f"\n• 💡 Tip: {recommendation}"
        else:
            analysis = f"\n\n🔍 My analysis:\n• 💡 Tip: {recommendation}"

    return base_msg + (analysis or "")

async def format_video_response(result: dict, is_last: bool,
                                tone: str = "neutral", is_correct: bool = True,
                                recipe_step: str = "") -> str:
    video_data = result.get("video", {})
    fusion_data = result.get("fusion", {})
    report_data = fusion_data.get("report", {}) or result.get("photo", {})

    video_action = video_data.get("top1") or video_data.get("action") or None
    main_food = report_data.get("photo_top1") or report_data.get("food")
    main_doneness = report_data.get("photo_doneness") or report_data.get("doneness")
    main_container = report_data.get("photo_container") or report_data.get("container")

    template = _get_template(tone, is_correct, is_last)
    if template:
        try:
            base_msg = template.format(
                action=video_action or "cooking",
                food=main_food or "unknown dish",
                doneness=main_doneness or "unknown doneness",
                container=main_container or "unknown container",
                recipe_step=recipe_step
            )
        except KeyError:
            logger.info("Error during template matching")
            base_msg = template
    else:
        if is_last:
            base_msg = f"🎉 Excellent! Your {main_food or 'dish'} looks complete!"
        else:
            base_msg = f"👨‍🍳 Great progress! I see you're {video_action or 'cooking'}."

    analysis = _format_analysis_lines([
        ("🎬 Action", video_action),
        ("🍽️ Main food", main_food),
        ("🔥 Doneness", main_doneness),
        ("🍳 Container", main_container),
    ])

    return base_msg + (analysis or "")


def log_inference_result(result: dict, media_type: str):
    if not isinstance(result, dict):
        logger.info(f"[SUMMARY] {media_type.title()}: Invalid result format")
        return

    result_type = result.get("type")

    if result_type == "image":
        photo_data = result.get("photo", {}) or result.get("fusion", {}).get("report", {})
        food = photo_data.get("food") or photo_data.get("photo_top1")
        doneness = photo_data.get("doneness") or photo_data.get("photo_doneness")
        container = photo_data.get("container") or photo_data.get("photo_container")
        recommendation = photo_data.get("recommendation") or photo_data.get("photo_recommendation", "")

        fields = []
        if not _is_unknown(food):
            fields.append(f"food={food}")
        if not _is_unknown(doneness):
            fields.append(f"doneness={doneness}")
        if not _is_unknown(container):
            fields.append(f"container={container}")
        if recommendation and not _is_unknown(recommendation):
            fields.append(f"recommendation={recommendation[:100]}...")

        logger.info(f"[SUMMARY] Photo: " + (", ".join(fields) if fields else "no useful fields found"))

    elif result_type == "video":
        video_data = result.get("video", {})
        fusion_data = result.get("fusion", {})
        report_data = fusion_data.get("report", {}) or result.get("photo", {})

        action = video_data.get("top1") or video_data.get("action")
        main_food = report_data.get("photo_top1") or report_data.get("food")
        doneness = report_data.get("photo_doneness") or report_data.get("doneness")
        container = report_data.get("photo_container") or report_data.get("container")

        fields = []
        if action:
            fields.append(f"action={action}")
        if not _is_unknown(main_food):
            fields.append(f"main_food={main_food}")
        if not _is_unknown(doneness):
            fields.append(f"doneness={doneness}")
        if not _is_unknown(container):
            fields.append(f"container={container}")

        logger.info(f"[SUMMARY] Video: " + (", ".join(fields) if fields else "no useful fields found"))

        frames = result.get("photo_frames", [])
        if frames:
            first_frame = frames[0]
            ff_food = first_frame.get("food")
            ff_doneness = first_frame.get("doneness")
            info = []
            if not _is_unknown(ff_food):
                info.append(f"food={ff_food}")
            if not _is_unknown(ff_doneness):
                info.append(f"doneness={ff_doneness}")
            if info:
                logger.info(f"[SUMMARY] First frame: " + ", ".join(info))


# --- Handlers ---
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Hi there! 👋 I'm your kitchen assistant bot.\nChoose who will cook with you:",
        reply_markup=mode_keyboard()
    )

@dp.message(F.text.in_(list(RECIPE_LIST.keys())))
async def choose_recipe(message: Message):
    user_id = message.from_user.id
    user_data.setdefault(user_id, {})
    if user_id not in user_data:
        # await message.answer("Please choose a mode first! /start")
        # return
        user_data[user_id] = {}

    recipe_key = RECIPE_LIST[message.text]
    user_data[user_id]["recipe"] = recipe_key
    user_data[user_id]["step"] = 0
    user_data[user_id]["cooking"] = True

    recipe = recipes[recipe_key]
    await message.answer(
        f"🍽 {recipe['name']}\n\n🕒 Time: {recipe['time']}\n🛒 Ingredients: {', '.join(recipe['ingredients'])}\n\n"
        f"Ready? Let's start with the first step!",
        reply_markup=actions_keyboard()
    )

    await send_next_step(message.chat.id, user_id)

@dp.message(F.text.in_(["👨‍🍳 None of them", "👵 Sweet Grandma", "🍳 Gordon Ramsay"]))
async def choose_initial_tone(message: Message):
    user_id = message.from_user.id

    initial_tone = {
        "🍳 Gordon Ramsay": "gordon",
        "👵 Sweet Grandma": "grandma",
        "👨‍🍳 None of them": "neutral"
    }[message.text]

    user_data[user_id] = {"initial_tone": initial_tone}

    await message.answer(
        f"Сool! Your chosen mode: {message.text}\n"
        "Which recipe we will cook together? Choose:",
        reply_markup=recipes_keyboard()
    )

async def send_next_step(chat_id, user_id):
    data = user_data.get(user_id)
    if not data or not data.get("recipe"):
        return

    recipe = recipes[data["recipe"]]
    steps = recipe["steps"]
    current_step_idx = data["step"]

    if current_step_idx >= len(steps): #all done
        if data.get("waiting_feedback", False):
            return
        user_data[user_id]["cooking"] = False
        user_data[user_id]["step"] = 0

        await bot.send_message(
            chat_id,
            "Done! Let's cook something else?",
            reply_markup=recipes_keyboard()
        )
        return

    step_text = steps[current_step_idx]

    await bot.send_message(chat_id, f"👨‍🍳 Step {current_step_idx + 1}: {step_text}\n\n📸 Send me a photo or video of your result!")

# --- Media Handlers ---
@dp.message(F.photo)
async def handle_photo(message: Message):
    user_id = message.from_user.id
    data = user_data.get(user_id)

    if not data or not data.get("cooking"):
        await message.answer("We're not cooking yet! Please choose a dish 😊", reply_markup=recipes_keyboard())
        return

    import tempfile
    from inference.unified_inference import run_inference
    photo = message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        file = await bot.download(photo, destination=tmp)
        tmp_path = tmp.name

    await bot.send_chat_action(message.chat.id, "typing")
    await message.answer("Let me see what you got there...")

    # await message.react("👀")

    try:
        result = run_inference(tmp_path)
        logger.info(f"Photo inference summary: type={result.get('type')}, food={result.get('photo', {}).get('food')}, doneness={result.get('photo', {}).get('doneness')}")
    except Exception as e:
        logger.exception(f"Photo inference failed: {e}")
        await message.answer("Error during photo analysis. Try again.")
        return
    finally:
        os.remove(tmp_path)

    msg = await format_inference_response(result, user_id)

    if not msg:
        msg = "Error - try another photo."

    feedback_text = "\n\nDo you like tone of this conversation?"
    await message.answer(msg + feedback_text, reply_markup=feedback_keyboard())

    log_inference_result(result, "photo")

    user_data[user_id]["waiting_feedback"] = True
    user_data[user_id]["last_result"] = result

@dp.message(F.video | F.video_note)
async def handle_video(message: Message):
    user_id = message.from_user.id
    data = user_data.get(user_id)

    if not data or not data.get("cooking"):
        await message.answer("We're not cooking yet! Please choose a dish 😊", reply_markup=recipes_keyboard())
        return

    import tempfile
    from inference.unified_inference import run_inference
    video = message.video or message.video_note
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file = await bot.download(video, destination=tmp)
        tmp_path = tmp.name

    await bot.send_chat_action(message.chat.id, "typing")
    await message.answer("Let me see what you got there...")

    # await message.react("👀")

    try:
        result = run_inference(tmp_path)
        logger.info(f"Video inference summary: type={result.get('type')}, top1={result.get('video', {}).get('top1')}, fusion_generated={result.get('fusion', {}).get('generated')}")
    except Exception as e:
        logger.exception(f"Video inference failed: {e}")
        await message.answer("Error during analysis, try again.")
        return
    finally:
        os.remove(tmp_path)

    msg = await format_inference_response(result, user_id)
    if not msg:
        msg = "Error - try another video."

    feedback_text = "\n\ndo you like tone of this conversation?"
    await message.answer(msg + feedback_text, reply_markup=feedback_keyboard())

    log_inference_result(result, "video")

    user_data[user_id]["waiting_feedback"] = True
    user_data[user_id]["last_result"] = result

@dp.message(F.text == "📖 View Recipe")
async def show_recipe(message: Message):
    user_id = message.from_user.id
    data = user_data.get(user_id)
    if not data or not data.get("recipe"):
        await message.answer("You haven’t chosen a dish yet! 🍲")
        return

    recipe = recipes[data["recipe"]]
    text = (
        f"🍽 {recipe['name']}\n\n"
        f"🕒 Cooking Time: {recipe['time']}\n"
        f"🛒 Ingredients: {', '.join(recipe['ingredients'])}\n\n"
        f"👨‍🍳 Steps:\n" + "\n".join(f'{i+1}. {step}' for i, step in enumerate(recipe['steps']))
    )
    await message.answer(text, reply_markup=actions_keyboard())

@dp.message(F.text == "🍲 Cook a New Dish")
async def new_recipe(message: Message):
    await message.answer("Choose a new dish:", reply_markup=recipes_keyboard())

@dp.message(F.text == "✅ Finish Cooking")
async def finish_cooking(message: Message):
    user_data.pop(message.from_user.id, None)
    await message.answer("Cooking finished! 👏 Come back when you’re hungry again!", reply_markup=recipes_keyboard())

@dp.callback_query(F.data.startswith("feedback_"))
async def feedback_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    data = user_data.get(user_id)
    choice = callback.data.split("_")[1]

    if not data:
        await callback.answer()
        return

    # Remove feedback buttons
    await callback.message.edit_reply_markup()
    user_data[user_id]["last_feedback"] = choice
    user_data[user_id]["waiting_feedback"] = False

    reward = 1 if choice == "like" else -1

    bandit_policy.update(
        user_id=user_id,
        state=data["last_state"],
        action=data["last_tone"],
        reward=reward
    )

    # Thank for feedback
    if choice == "like":
        await callback.message.answer("Thanks for the feedback! ❤️ Glad you liked it 😄")
    else:
        await callback.message.answer("Thanks for your feedback! 👌 I’ll try to improve 😊")

    # Move to the next step
    if data.get("cooking"):
        user_data[user_id]["step"] += 1
        await asyncio.sleep(1)
        await send_next_step(callback.message.chat.id, user_id)

    await callback.answer()

# --- General text handler ---
@dp.message(F.text)
async def handle_text_only(message: Message):
    user_id = message.from_user.id
    data = user_data.get(user_id)

    if data and data.get("cooking"):
        await message.answer("I'm waiting for a photo or video of your result. Please send it!", reply_markup=actions_keyboard())
        return

    await message.answer("Choose an action from the menu 😊", reply_markup=actions_keyboard())

# --- Entry point ---
async def main():
    logger.info("✅ Bot is ready and polling for updates")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped")