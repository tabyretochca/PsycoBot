import os
from dotenv import load_dotenv
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models.gigachat import GigaChat

# Загрузка переменных из .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")

# Инициализация GigaChat
llm = GigaChat(
    credentials=GIGACHAT_TOKEN,
    verify_ssl_certs=False,  # Отключаем проверку SSL для упрощения (в продакшене лучше включить)
    model="GigaChat",  # Базовая модель GigaChat
)

# Шаблон для психологической поддержки
support_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    Ты эмпатичный психолог, который помогает людям справляться с эмоциями и проблемами.
    Вот история нашего разговора: {history}
    Пользователь написал: {input}
    Ответь так, чтобы поддержать его, дать совет или задать уточняющий вопрос.
    """
)

sentiment_template = PromptTemplate(
    input_variables=["input"],
    template="Определи тональность этого текста (позитивная, негативная, нейтральная): {input}"
)

# Создание цепочки с памятью
memory = ConversationBufferMemory(memory_key="history", input_key="input")
support_chain = LLMChain(llm=llm, prompt=support_template, memory=memory)
sentiment_chain = LLMChain(llm=llm, prompt=sentiment_template)

# Команда /start
def start(update, context):
    welcome_message = (
        "Привет! Я твой персональный бот-психолог. "
        "Расскажи мне, что тебя беспокоит, или просто поделись своими мыслями — я здесь, чтобы помочь!"
    )
    update.message.reply_text(welcome_message)

def reset(update, context):
    memory.clear()
    update.message.reply_text("История разговора очищена. Давай начнём заново!")

def help_command(update, context):
    help_text = (
        "Я твой бот-психолог! Вот что я умею:\n"
        "/start - Начать общение\n"
        "/reset - Очистить историю разговора\n"
        "/help - Показать эту справку\n"
        "/mood - Оценить твоё настроение\n"
        "Просто пиши мне свои мысли, и я постараюсь помочь!"
    )
    update.message.reply_text(help_text)

def mood(update, context):
    update.message.reply_text("Как оценишь своё настроение от 1 до 10? Напиши число или опиши словами.")

# Обработка текстовых сообщений
def handle_message(update, context):
    user_input = update.message.text
    try:
        response = support_chain.run(input=user_input)
        update.message.reply_text(response)
    except Exception as e:
        update.message.reply_text(f"Что-то пошло не так: {str(e)}. Попробуй ещё раз!")

# Основная функция
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_handler(CommandHandler("reset", reset))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("mood", mood))

    print("Бот запущен!")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()