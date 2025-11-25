from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.filters import CommandStart
from dotenv import load_dotenv
import os
import asyncio
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

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

# --- Keyboards ---
def mode_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🍳 Gordon Ramsay Mode"), KeyboardButton(text="👵 Sweet Grandma Mode")]
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

# --- Handlers ---
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Hi there! 👋 I'm your kitchen assistant bot.\nChoose who will cook with you:",
        reply_markup=mode_keyboard()
    )

@dp.message(F.text.in_(["🍳 Gordon Ramsay Mode", "👵 Sweet Grandma Mode"]))
async def choose_mode(message: Message):
    user_data[message.from_user.id] = {"mode": message.text, "recipe": None, "step": 0, "cooking": False}
    await message.answer(
        f"Awesome! Your mentor is {message.text}.\nNow choose a dish:",
        reply_markup=recipes_keyboard()
    )

@dp.message(F.text.in_(list(RECIPE_LIST.keys())))
async def choose_recipe(message: Message):
    user_id = message.from_user.id
    if user_id not in user_data:
        await message.answer("Please choose a mode first! /start")
        return

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
        await bot.send_message(chat_id, "🎉 Dish completed! Great job, chef!", reply_markup=mode_keyboard())
        return

    step_text = steps[current_step]
    await bot.send_message(chat_id, f"👨‍🍳 Step {current_step + 1}: {step_text}\n\n📸 Send me a photo or video of your result!")

# --- Media Handlers ---
@dp.message(F.photo)
async def handle_photo(message: Message):
    user_id = message.from_user.id
    data = user_data.get(user_id)

    if not data or not data.get("cooking"):
        await message.answer("We're not cooking yet! Please choose a dish 😊", reply_markup=mode_keyboard())
        return

    import tempfile
    from inference.unified_inference import run_inference
    import logging
    logger = logging.getLogger("bchef.bot")
    # Сохраняем фото во временный файл
    photo = message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        file = await bot.download(photo, destination=tmp)
        tmp_path = tmp.name

    try:
        result = run_inference(tmp_path)
        logger.info(f"Photo inference result: {result}")
    except Exception as e:
        logger.exception(f"Photo inference failed: {e}")
        await message.answer("Ошибка при анализе фото. Попробуйте еще раз.")
        return
    finally:
        os.remove(tmp_path)

    # Формируем красивый ответ
    msg = None
    if isinstance(result, dict) and ("photo" in result or "food" in result):
        photo = result.get("photo", result)
        food = photo.get("food") or photo.get("top1") or "?"
        doneness = photo.get("doneness", "?")
        recommendation = photo.get("recommendation", "")
        # Определяем последний ли шаг
        user_recipe = user_data.get(user_id, {}).get("recipe")
        user_step = user_data.get(user_id, {}).get("step", 0)
        steps = recipes.get(user_recipe, {}).get("steps", [])
        # Проверяем: если cooking==False, значит это был последний шаг
        is_last = not user_data.get(user_id, {}).get("cooking", True) or (steps and user_step >= len(steps))
        if is_last:
            ending = "Recipe completed, bon appétit!"
        else:
            ending = "Let's move to the next step!"
        msg = f"I see {food}.\n{ending}"
    if not msg:
        msg = "Error - try another photo."

    await message.answer(msg, reply_markup=feedback_keyboard())
    # Короткий лог: только ключевые поля
    if isinstance(result, dict):
        if result.get("type") == "image" and "photo" in result:
            photo = result["photo"]
            logger.info(f"[SUMMARY] Photo: food={photo.get('food')}, doneness={photo.get('doneness')}, container={photo.get('container')}, recommendation={photo.get('recommendation')}")
        elif result.get("type") == "video":
            video = result.get("video", {})
            logger.info(f"[SUMMARY] Video: top1={video.get('top1')}, scores={video.get('scores')}")
            frames = result.get("photo_frames", [])
            if frames:
                first = frames[0]
                logger.info(f"[SUMMARY] First frame: food={first.get('food')}, doneness={first.get('doneness')}, recommendation={first.get('recommendation')}")
        else:
            logger.info(f"[SUMMARY] Inference: {str(result)[:200]}")
    else:
        logger.info(f"[SUMMARY] Inference: {str(result)[:200]}")
    # (удалено, теперь только короткий лог)
    user_data[user_id]["waiting_feedback"] = True

@dp.message(F.video | F.video_note)
async def handle_video(message: Message):
    user_id = message.from_user.id
    data = user_data.get(user_id)

    if not data or not data.get("cooking"):
        await message.answer("We're not cooking yet! Please choose a dish 😊", reply_markup=mode_keyboard())
        return

    import tempfile
    from inference.unified_inference import run_inference
    import logging
    logger = logging.getLogger("bchef.bot")
    # Сохраняем видео во временный файл
    video = message.video or message.video_note
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file = await bot.download(video, destination=tmp)
        tmp_path = tmp.name

    try:
        result = run_inference(tmp_path)
        logger.info(f"Video inference result: {result}")
    except Exception as e:
        logger.exception(f"Video inference failed: {e}")
        await message.answer("Ошибка при анализе видео. Попробуйте еще раз.")
        return
    finally:
        os.remove(tmp_path)

    # Формируем красивый ответ
    msg = None
    if isinstance(result, dict) and result.get("type") == "video":
        video = result.get("video", {})
        top1 = video.get("top1", "?")
        # Берём первый кадр фото для примера
        photo_frames = result.get("photo_frames", [])
        if photo_frames:
            photo = photo_frames[0]
            food = photo.get("food") or photo.get("top1") or "?"
            doneness = photo.get("doneness", "?")
            recommendation = photo.get("recommendation", "")
            user_recipe = user_data.get(user_id, {}).get("recipe")
            user_step = user_data.get(user_id, {}).get("step", 0)
            steps = recipes.get(user_recipe, {}).get("steps", [])
            is_last = not user_data.get(user_id, {}).get("cooking", True) or (steps and user_step >= len(steps))
            if is_last:
                ending = "Recipe completed, bon appétit!"
            else:
                ending = "Let's move to the next step!"
            msg = f"Great! You sent a video with {top1}. \n{ending}"
    if not msg:
        msg = "Error - try another video."

    await message.answer(msg, reply_markup=feedback_keyboard())
    # Короткий лог: только ключевые поля
    if isinstance(result, dict):
        if result.get("type") == "video":
            video = result.get("video", {})
            logger.info(f"[SUMMARY] Video: top1={video.get('top1')}, scores={video.get('scores')}")
            frames = result.get("photo_frames", [])
            if frames:
                first = frames[0]
                logger.info(f"[SUMMARY] First frame: food={first.get('food')}, doneness={first.get('doneness')}, recommendation={first.get('recommendation')}")
        elif result.get("type") == "image" and "photo" in result:
            photo = result["photo"]
            logger.info(f"[SUMMARY] Photo: food={photo.get('food')}, doneness={photo.get('doneness')}, container={photo.get('container')}, recommendation={photo.get('recommendation')}")
        else:
            logger.info(f"[SUMMARY] Inference: {str(result)[:200]}")
    else:
        logger.info(f"[SUMMARY] Inference: {str(result)[:200]}")
    # (удалено, теперь только короткий лог)
    user_data[user_id]["waiting_feedback"] = True

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
    await message.answer("Cooking finished! 👏 Come back when you’re hungry again!", reply_markup=mode_keyboard())

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