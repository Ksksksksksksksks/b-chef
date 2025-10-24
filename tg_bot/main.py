from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.filters import CommandStart
import asyncio
import json

bot = Bot(token="8198073291:AAHkwGVrX2TWllCBSPrfGP978B6BH0WnJwk")
dp = Dispatcher()

# Загружаем рецепты из JSON
with open("recipes.json", "r", encoding="utf-8") as f:
    recipes = json.load(f)

# Список всех рецептов для фильтров и клавиатуры
RECIPE_LIST = {
    "🥚 Яичница": "яичница",
    "🍝 Макароны": "макароны",
    "🍚 Рис": "рис",
    "🌾 Гречка": "гречка",
    "🥚 Варёные яйца": "яйца",
    "🥩 Жареное мясо": "жареное мясо",
    "🍖 Тушёное мясо": "тушёное мясо",
    "🥔 Жареная картошка": "жареная картошка"
}

# Временное "хранилище" для выбранных опций пользователей
user_data = {}

# --- Клавиатуры ---
def mode_keyboard():
    kb = [
        [KeyboardButton(text="🍳 Суровый Рамзи"), KeyboardButton(text="👵 Милая бабушка")]
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def recipes_keyboard():
    kb = [
        [KeyboardButton(text="🥚 Яичница"), KeyboardButton(text="🍝 Макароны")],
        [KeyboardButton(text="🍚 Рис"), KeyboardButton(text="🌾 Гречка")],
        [KeyboardButton(text="🥚 Варёные яйца"), KeyboardButton(text="🥔 Жареная картошка")],
        [KeyboardButton(text="🥩 Жареное мясо"), KeyboardButton(text="🍖 Тушёное мясо")]
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def actions_keyboard():
    kb = [
        [KeyboardButton(text="📖 Посмотреть рецепт")],
        [KeyboardButton(text="🍲 Приготовить новое блюдо")],
        [KeyboardButton(text="✅ Завершить готовку")]
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

# --- Inline-клавиатура для оценки ---
def feedback_keyboard():
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="👍 Лайк", callback_data="feedback_like"),
             InlineKeyboardButton(text="👎 Дизлайк", callback_data="feedback_dislike")]
        ]
    )
    return kb

# --- Обработчики ---
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! 👋 Я бот-помощник на кухне.\nВыбери, кто будет с тобой готовить:",
        reply_markup=mode_keyboard()
    )

@dp.message(F.text.in_(["🍳 Суровый Рамзи", "👵 Милая бабушка"]))
async def choose_mode(message: Message):
    user_data[message.from_user.id] = {"mode": message.text, "recipe": None}
    await message.answer(
        f"Отлично! Твой наставник — {message.text}.\nТеперь выбери блюдо:",
        reply_markup=recipes_keyboard()
    )

@dp.message(F.text.in_(list(RECIPE_LIST.keys())))
async def choose_recipe(message: Message):
    user_id = message.from_user.id
    if user_id not in user_data:
        await message.answer("Сначала выбери режим общения! /start")
        return

    recipe_name = RECIPE_LIST[message.text]
    user_data[user_id]["recipe"] = recipe_name
    recipe = recipes[recipe_name]

    text = (
        f"🍽 {recipe['name']}\n\n"
        f"🕒 Время приготовления: {recipe['time']}\n"
        f"🛒 Продукты: {', '.join(recipe['ingredients'])}\n\n"
        f"👨‍🍳 Шаги:\n" + "\n".join(f'{i+1}. {step}' for i, step in enumerate(recipe['steps'])) +
        "\n\n📸 Жду твои фото и видео процесса готовки!"
    )

    await message.answer(text, reply_markup=actions_keyboard())

@dp.message(F.text == "📖 Посмотреть рецепт")
async def show_recipe(message: Message):
    user_id = message.from_user.id
    data = user_data.get(user_id)
    if not data or not data.get("recipe"):
        await message.answer("Ты ещё не выбрал блюдо! 🍲")
        return

    recipe = recipes[data["recipe"]]
    text = (
        f"🍽 {recipe['name']}\n\n"
        f"🕒 Время приготовления: {recipe['time']}\n"
        f"🛒 Продукты: {', '.join(recipe['ingredients'])}\n\n"
        f"👨‍🍳 Шаги:\n" + "\n".join(f'{i+1}. {step}' for i, step in enumerate(recipe['steps']))
    )
    await message.answer(text, reply_markup=actions_keyboard())

@dp.message(F.text == "🍲 Приготовить новое блюдо")
async def new_recipe(message: Message):
    await message.answer("Выбери новое блюдо:", reply_markup=recipes_keyboard())

@dp.message(F.text == "✅ Завершить готовку")
async def finish_cooking(message: Message):
    user_data.pop(message.from_user.id, None)
    await message.answer("Готовка завершена! 👏 Возвращайся, когда проголодаешься!", reply_markup=mode_keyboard())

# --- Обработка фото и видео с кнопкой оценки ---
@dp.message(F.video_note)
async def handle_video(message: Message):
    await message.reply(
        "Спасибо за видео! 🎥 У тебя хорошо получается, продолжай в том же духе!",
        reply_markup=feedback_keyboard()
    )

@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.reply(
        "Спасибо за фото! 📸 У тебя хорошо получается, продолжай в том же духе!",
        reply_markup=feedback_keyboard()
    )

# --- Обработка выбора оценки ---
@dp.callback_query(F.data.startswith("feedback_"))
async def feedback_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    choice = callback.data.split("_")[1]  # "like" или "dislike"

    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id]["last_feedback"] = choice

    await callback.message.edit_reply_markup()  # убираем кнопки после выбора
    if choice == "like":
        await callback.message.answer("Спасибо за лайк! 😊")
    else:
        await callback.message.answer("Спасибо за обратную связь! 👍")

    await callback.answer()  # убираем "часики" на кнопке

# --- Остальные сообщения ---
@dp.message()
async def handle_other(message: Message):
    await message.answer("Выбери действие из меню 😊", reply_markup=actions_keyboard())

# --- Точка входа ---
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот выключен")
