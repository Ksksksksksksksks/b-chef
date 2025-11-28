from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.filters import CommandStart
from dotenv import load_dotenv
import os
import sys
import asyncio
import json
import logging

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
    "🍚 Rice": "rice",
    "🌾 Buckwheat": "buckwheat",
    "🥚 Boiled Eggs": "boiled eggs",
    "🥩 Fried Meat": "fried meat",
    "🍖 Stewed Meat": "stewed meat",
    "🥔 Fried Potatoes": "fried potatoes"
}

user_data = {}

# rl police init
bandit_policy = BanditPolicy(path="qtable.json")

# tones dir
TONES_DIR = os.path.join(BASE_DIR, "rl/tones")

_template_cache = {}


def _load_tone_templates(tone: str):
    if tone in _template_cache:
        return _template_cache[tone]

    path = os.path.join(TONES_DIR, f"{tone}.txt")
    if not os.path.exists(path):
        path = os.path.join(TONES_DIR, "neutral.txt")
        if not os.path.exists(path):
            return None

    templates = {"correct": "", "incorrect": "", "correct_last": "", "incorrect_last": ""}
    current_key = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current_key = line[1:-1]
            elif current_key and line:
                templates[current_key] += line + "\n"

    for key in templates:
        templates[key] = templates[key].strip()

    _template_cache[tone] = templates
    return templates


def _get_template(tone: str, is_correct: bool, is_last: bool) -> str:
    templates = _load_tone_templates(tone)
    if not templates:
        return None

    key = ("correct" if is_correct else "incorrect") + ("_last" if is_last else "")
    return templates.get(key) or templates.get("correct" if is_correct else "incorrect", "")

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
            [KeyboardButton(text="🥚 Fried Eggs"), KeyboardButton(text="🍝 Pasta")],
            [KeyboardButton(text="🍚 Rice"), KeyboardButton(text="🌾 Buckwheat")],
            [KeyboardButton(text="🥚 Boiled Eggs"), KeyboardButton(text="🥔 Fried Potatoes")],
            [KeyboardButton(text="🥩 Fried Meat"), KeyboardButton(text="🍖 Stewed Meat")]
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

def determine_state(result, user_data):
    current_step = user_data.get("step", 0)
    recipe = recipes.get(user_data.get("recipe", ""), {})
    step_text = recipe.get("steps", [])[current_step] if current_step < len(recipe.get("steps", [])) else ""

    if "fry" in step_text.lower() and result.get("photo", {}).get("doneness") in ["medium", "well done"]:
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
    state = determine_state(result, user_data.get(user_id, {}))
    is_correct = state == 1
    tone = bandit_policy.choose_action(user_id, state, user_data)

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


async def format_photo_response(result: dict, is_last: bool,
                                tone: str = "neutral", is_correct: bool = True,
                                recipe_step: str = "") -> str:
    photo_data = result.get("photo", {})

    food = photo_data.get("food", "unknown dish")
    doneness = photo_data.get("doneness", "unknown doneness")
    container = photo_data.get("container", "unknown container")
    recommendation = photo_data.get("recommendation", "")

    template = _get_template(tone, is_correct, is_last)
    if template:
        try:
            base_msg = template.format(
                food=food,
                doneness=doneness,
                container=container,
                recommendation=recommendation,
                recipe_step=recipe_step
            )
        except KeyError:
            logger.info("Error during template matching")
            base_msg = template
    else:
        if is_last:
            base_msg = f"🎉 Perfect! I see your {food} is ready!"
        else:
            base_msg = f"👨‍🍳 Great! I see you're working on {food}."

    analysis_details = f"\n\n🔍 My analysis:\n"
    analysis_details += f"• 🍽️ Food: {food}\n"
    analysis_details += f"• 🔥 Doneness: {doneness}\n"
    analysis_details += f"• 🍳 Container: {container}\n"

    if recommendation and recommendation != "Unknown food type.":
        analysis_details += f"• 💡 Tip: {recommendation}"

    return base_msg + analysis_details


async def format_video_response(result: dict, is_last: bool,
                                tone: str = "neutral", is_correct: bool = True,
                                recipe_step: str = "") -> str:
    video_data = result.get("video", {})
    fusion_data = result.get("fusion", {})
    report_data = fusion_data.get("report", {})

    video_action = video_data.get("top1", "cooking")
    main_food = report_data.get("photo_top1", "unknown dish")
    main_doneness = report_data.get("photo_doneness", "unknown doneness")
    main_container = report_data.get("photo_container", "unknown container")

    template = _get_template(tone, is_correct, is_last)
    if template:
        try:
            base_msg = template.format(
                action=video_action,
                food=main_food,
                doneness=main_doneness,
                container=main_container,
                recipe_step=recipe_step
            )
        except KeyError:
            logger.info("Error during template matching")
            base_msg = template
    else:
        if is_last:
            base_msg = f"🎉 Excellent! Your {main_food} looks complete!"
        else:
            base_msg = f"👨‍🍳 Great progress! I see you're {video_action}."

    analysis_details = f"\n\n🔍 My analysis:\n"
    analysis_details += f"• 🎬 Action: {video_action}\n"
    analysis_details += f"• 🍽️ Main food: {main_food}\n"
    analysis_details += f"• 🔥 Doneness: {main_doneness}\n"
    analysis_details += f"• 🍳 Container: {main_container}\n"

    return base_msg + analysis_details


def log_inference_result(result: dict, media_type: str):

    if not isinstance(result, dict):
        logger.info(f"[SUMMARY] {media_type.title()}: Invalid result format")
        return

    result_type = result.get("type")

    if result_type == "image":
        photo_data = result.get("photo", {})
        logger.info(f"[SUMMARY] Photo: food={photo_data.get('food')}, "
                    f"doneness={photo_data.get('doneness')}, "
                    f"container={photo_data.get('container')}, "
                    f"recommendation={photo_data.get('recommendation', '')[:100]}...")

    elif result_type == "video":
        video_data = result.get("video", {})
        fusion_data = result.get("fusion", {})
        report_data = fusion_data.get("report", {})

        logger.info(f"[SUMMARY] Video: action={video_data.get('top1')}, "
                    f"main_food={report_data.get('photo_top1')}, "
                    f"doneness={report_data.get('photo_doneness')}, "
                    f"container={report_data.get('photo_container')}")

        frames = result.get("photo_frames", [])
        if frames:
            first_frame = frames[0]
            logger.info(f"[SUMMARY] First frame: food={first_frame.get('food')}, "
                        f"doneness={first_frame.get('doneness')}")


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
    current_step = data["step"]

    if current_step >= len(steps):
        user_data[user_id]["cooking"] = False
        user_data[user_id]["step"] = 0
        await bot.send_message(chat_id, "🎉 Dish completed! Great job, chef!", reply_markup=recipes_keyboard())
        return

    step_text = steps[current_step]
    await bot.send_message(chat_id, f"👨‍🍳 Step {current_step + 1}: {step_text}\n\n📸 Send me a photo or video of your result!")

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
    import logging
    logger = logging.getLogger("bchef.bot")
    photo = message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        file = await bot.download(photo, destination=tmp)
        tmp_path = tmp.name

    await bot.send_chat_action(message.chat.id, "typing")
    await message.answer("Let me see what you got there...")

    # await message.react("👀")

    try:
        result = run_inference(tmp_path)
        logger.info(f"Photo inference result: {result}")
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
    import logging
    logger = logging.getLogger("bchef.bot")
    video = message.video or message.video_note
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file = await bot.download(video, destination=tmp)
        tmp_path = tmp.name

    await bot.send_chat_action(message.chat.id, "typing")
    await message.answer("Let me see what you got there...")

    # await message.react("👀")

    try:
        result = run_inference(tmp_path)
        logger.info(f"Video inference result: {result}")
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
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped")