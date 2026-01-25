import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from tasks import process_ros2_query

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот по ROS 2 Humble. Задавай вопрос, я поставлю его в очередь.")

@dp.message()
async def handle_message(message: types.Message):
    # Отправляем задачу в Celery
    # delay() - это асинхронная отправка в Redis
    task = process_ros2_query.delay(message.chat.id, message.text)
    
    await message.answer(
        f"⏳ Вопрос принят в обработку!\nID задачи: `{task.id}`\n\n"
        "Так как запрос передается в очередь для бесплатных клиентов, ответ может занять до 5 минут.",
        parse_mode="Markdown"
    )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())