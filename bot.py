import os
import re
import html
import asyncio
import logging
import random
import time
import json
from collections import deque

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

from dotenv import load_dotenv
from telegram import Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.error import Forbidden, BadRequest, NetworkError, TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

import fal_client
from qwen_model import generate_qwen_image, QwenImageGenerator, SIZE_MAP, ALLOWED_SIZES
from seedream_model import (
    generate_seedream_image,
    SeDreamImageGenerator,
    SEEDREAM_SIZE_MAP,
    SEEDREAM_ALLOWED_SIZES,
    generate_seedream_edit,
    convert_user_edit_size_to_api,
    SEEDREAM_EDIT_ALLOWED_SIZES,
)


BOT_VERSION = "2.2"
BOT_USERNAME = "@BabakaEbaka_bot"  # Имя бота для ссылок

# Модели
MODEL_QWEN = "qwen"
MODEL_SEEDREAM = "seedream"

# Информация о моделях
MODEL_INFO = {
    MODEL_QWEN: {
        "name": "Qwen-Image",
        "display_name": "Qwen-Image", 
        "description": "Стабильная и быстрая модель для повседневного творчества. Хорошо понимает русский язык и создает качественные изображения в HD разрешении.",
        "default_size": "Квадрат 2K",
        "size_map": SIZE_MAP,
        "allowed_sizes": ALLOWED_SIZES
    },
    MODEL_SEEDREAM: {
        "name": "SeDream v4",
        "display_name": "SeDream 4.0",
        "description": "Продвинутая модель нового поколения от Bytedance. Превосходное качество до 4K, глубокое понимание стилей, улучшенная детализация и креативность.",
        "default_size": "Квадрат 2K", 
        "size_map": SEEDREAM_SIZE_MAP,
        "allowed_sizes": SEEDREAM_ALLOWED_SIZES
    }
}

# Время запуска бота для игнорирования старых сообщений
BOT_START_TIME = 0

# Отслеживание активных генераций по пользователям
ACTIVE_GENERATIONS: Dict[int, bool] = {}

# Лимиты генерации: до 38 изображений на пользователя за последний час
USER_HOURLY_WINDOW_SECONDS = 3600
USER_HOURLY_LIMIT = 38
# user_id -> deque[timestamps]
USER_GEN_TIMESTAMPS: Dict[int, deque] = {}

def _prune_and_count_user_window(user_id: int, now_ts: float) -> int:
    q = USER_GEN_TIMESTAMPS.get(user_id)
    if q is None:
        q = deque()
        USER_GEN_TIMESTAMPS[user_id] = q
    # удалить старые метки (старше часа)
    cutoff = now_ts - USER_HOURLY_WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()
    return len(q)

def _can_user_generate(user_id: int) -> Tuple[bool, int]:
    now_ts = time.time()
    count = _prune_and_count_user_window(user_id, now_ts)
    return (count < USER_HOURLY_LIMIT), (USER_HOURLY_LIMIT - count)

def _mark_user_generated(user_id: int, images_count: int = 1) -> None:
    now_ts = time.time()
    q = USER_GEN_TIMESTAMPS.get(user_id)
    if q is None:
        q = deque()
        USER_GEN_TIMESTAMPS[user_id] = q
    for _ in range(max(1, images_count)):
        q.append(now_ts)

# Файл для сохранения статистики
STATS_FILE = "bot_stats.json"
# Файл для сохранения общей статистики бота
BOT_ANALYTICS_FILE = "bot_analytics.json"


# Configure beautiful and structured logging
class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Configure logging with custom formatter
logging.basicConfig(
    level=logging.WARNING,  # Set higher level to suppress noise
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING) 
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Create our bot logger with colored formatter
logger = logging.getLogger("qwen-bot")
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
))
logger.handlers.clear()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)  # Our bot logs at INFO level


# Размеры теперь импортируются из qwen_model
ImageSizeValue = Union[str, Dict[str, int]]

# Telegram caption hard limit for media messages
CAPTION_MAX_LEN = 1024


def _split_caption(caption: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Ensure caption <= Telegram's limit. Returns (safe_caption, overflow_text).
    If truncated, overflow_text contains the full original caption to be sent as a separate message.
    """
    if not caption:
        return caption, None
    if len(caption) <= CAPTION_MAX_LEN:
        return caption, None
    safe_caption = caption[: CAPTION_MAX_LEN - 1] + "…"
    return safe_caption, caption


# --- Reply helpers to ensure thread-safe messaging in topic-enabled groups ---
async def _reply_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None, parse_mode: Optional[str] = None):
    """Reply in-topic using reply_to_message_id first; fallback to message_thread_id; then plain send."""
    chat_id = update.effective_chat.id
    message_id = update.effective_message.message_id if update.effective_message else None
    thread_id = getattr(update.effective_message, "message_thread_id", None) if update.effective_message else None
    # 1) Try reply_to only
    if message_id is not None:
        try:
            return await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                reply_to_message_id=message_id,
            )
        except BadRequest as e:
            # If replied message is gone or invalid, try thread id
            if "message thread not found" not in str(e).lower() and "replied message not found" not in str(e).lower():
                # fall through to thread id attempt below
                pass
    # 2) Try message_thread_id
    if thread_id is not None:
        try:
            return await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                message_thread_id=thread_id,
            )
        except BadRequest:
            pass
    # 3) Fallback: reply method
    if update.effective_message:
        try:
            return await update.effective_message.reply_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
        except BadRequest:
            pass
    # 4) Final fallback: plain send
    return await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup, parse_mode=parse_mode)


async def _reply_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, photo: str, caption: Optional[str] = None, reply_markup: Optional[InlineKeyboardMarkup] = None, parse_mode: Optional[str] = None):
    chat_id = update.effective_chat.id
    message_id = update.effective_message.message_id if update.effective_message else None
    thread_id = getattr(update.effective_message, "message_thread_id", None) if update.effective_message else None
    # 1) Try reply_to only
    if message_id is not None:
        try:
            return await context.bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=caption,
                reply_markup=reply_markup,
                reply_to_message_id=message_id,
                parse_mode=parse_mode,
            )
        except BadRequest:
            pass
    # 2) Try message_thread_id
    if thread_id is not None:
        try:
            return await context.bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=caption,
                reply_markup=reply_markup,
                message_thread_id=thread_id,
                parse_mode=parse_mode,
            )
        except BadRequest:
            pass
    # 3) Fallback reply method
    if update.effective_message:
        try:
            return await update.effective_message.reply_photo(photo=photo, caption=caption, reply_markup=reply_markup, parse_mode=parse_mode)
        except BadRequest:
            pass
    # 4) Final fallback
    return await context.bot.send_photo(chat_id=chat_id, photo=photo, caption=caption, reply_markup=reply_markup, parse_mode=parse_mode)


async def _reply_document(update: Update, context: ContextTypes.DEFAULT_TYPE, document: str, caption: Optional[str] = None, reply_markup: Optional[InlineKeyboardMarkup] = None, parse_mode: Optional[str] = None):
    chat_id = update.effective_chat.id
    message_id = update.effective_message.message_id if update.effective_message else None
    thread_id = getattr(update.effective_message, "message_thread_id", None) if update.effective_message else None
    # 1) Try reply_to only
    if message_id is not None:
        try:
            return await context.bot.send_document(
                chat_id=chat_id,
                document=document,
                caption=caption,
                reply_markup=reply_markup,
                reply_to_message_id=message_id,
                parse_mode=parse_mode,
            )
        except BadRequest:
            pass
    # 2) Try message_thread_id
    if thread_id is not None:
        try:
            return await context.bot.send_document(
                chat_id=chat_id,
                document=document,
                caption=caption,
                reply_markup=reply_markup,
                message_thread_id=thread_id,
                parse_mode=parse_mode,
            )
        except BadRequest:
            pass
    # 3) Fallback reply method
    if update.effective_message:
        try:
            return await update.effective_message.reply_document(document=document, caption=caption, reply_markup=reply_markup, parse_mode=parse_mode)
        except BadRequest:
            pass
    # 4) Final fallback
    return await context.bot.send_document(chat_id=chat_id, document=document, caption=caption, reply_markup=reply_markup, parse_mode=parse_mode)


async def _reply_media_group(update: Update, context: ContextTypes.DEFAULT_TYPE, media: List[InputMediaPhoto]):
    chat_id = update.effective_chat.id
    message_id = update.effective_message.message_id if update.effective_message else None
    thread_id = getattr(update.effective_message, "message_thread_id", None) if update.effective_message else None
    # 1) Try reply_to only (supported by Bot API)
    if message_id is not None:
        try:
            return await context.bot.send_media_group(chat_id=chat_id, media=media, reply_to_message_id=message_id)
        except BadRequest:
            pass
    # 2) Try message_thread_id
    if thread_id is not None:
        try:
            return await context.bot.send_media_group(chat_id=chat_id, media=media, message_thread_id=thread_id)
        except BadRequest:
            pass
    # 3) Fallback reply method
    if update.effective_message:
        try:
            return await update.effective_message.reply_media_group(media=media)
        except BadRequest:
            pass
    # 4) Final fallback
    return await context.bot.send_media_group(chat_id=chat_id, media=media)


async def _send_waiting(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Send a waiting/progress message in-thread when possible."""
    return await _reply_text(update, context, text)


async def _send_chat_action(context: ContextTypes.DEFAULT_TYPE, update: Update, action: ChatAction):
    """Send chat action without thread pinning to avoid BadRequest in some topic setups."""
    return await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=action)


async def _extract_image_urls(update: Update, context: ContextTypes.DEFAULT_TYPE) -> List[str]:
    """Извлекает до 10 URL изображений из текущего сообщения и/или ответа.
    Поддерживает фотографии и документы с типом image/*.
    """
    message = update.message if update.message else update.effective_message
    if not message:
        return []

    file_ids: List[str] = []

    def _collect_from_message(msg) -> None:
        if not msg:
            return
        # Photos
        if getattr(msg, "photo", None):
            # Берем наибольшее качество
            largest = max(msg.photo, key=lambda p: p.file_size or 0)
            file_ids.append(largest.file_id)
        # Image documents
        if getattr(msg, "document", None):
            doc = msg.document
            if doc and getattr(doc, "mime_type", "").startswith("image/"):
                file_ids.append(doc.file_id)

    _collect_from_message(message)
    # Если есть reply_to_message — берем из него
    _collect_from_message(getattr(message, "reply_to_message", None))

    urls: List[str] = []
    token = os.getenv("TELEGRAM_BOT_TOKEN") or getattr(context.bot, "token", None)
    for fid in file_ids[:10]:
        try:
            file = await context.bot.get_file(fid)
            file_path = getattr(file, "file_path", None)
            if file_path and token:
                # Проверяем, что file_path не содержит уже полный URL
                if file_path.startswith("https://"):
                    urls.append(file_path)
                else:
                    urls.append(f"https://api.telegram.org/file/bot{token}/{file_path}")
        except Exception as e:
            logger.warning(f"Failed to get file URL for {fid}: {e}")
            continue
    return urls


@dataclass
class SessionState:
    # Выбранная модель (по умолчанию SeDream)
    selected_model: str = MODEL_SEEDREAM
    
    # Настройки для Qwen
    qwen_image_size: str = "Квадрат 2K"
    qwen_enable_safety_checker: bool = True
    qwen_seed: Union[int, None] = None
    qwen_high_quality: bool = False
    
    # Настройки для SeDream  
    seedream_image_size: str = "Квадрат 2K"
    seedream_enable_safety_checker: bool = True  # По умолчанию теперь ВКЛ
    seedream_seed: Union[int, None] = None
    seedream_high_quality: bool = False
    seedream_4k_mode: bool = False  # Специальный 4K режим

    # Режим редактирования изображений (SeDream Edit)
    edit_mode_enabled: bool = False
    seedream_edit_image_size: str = "Авто 2K"  # auto_2K по умолчанию
    
    # Состояние ожидания промпта для генерации
    awaiting_generation_prompt: bool = False
    
    # Методы для получения текущих настроек
    @property
    def image_size(self) -> str:
        """Получить размер изображения для текущей модели"""
        if self.selected_model == MODEL_SEEDREAM:
            return "4K Квадрат" if self.seedream_4k_mode else self.seedream_image_size
        return self.qwen_image_size
    
    @property
    def enable_safety_checker(self) -> bool:
        """Получить настройку safety checker для текущей модели"""
        return self.seedream_enable_safety_checker if self.selected_model == MODEL_SEEDREAM else self.qwen_enable_safety_checker
    
    @property
    def seed(self) -> Union[int, None]:
        """Получить seed для текущей модели"""
        return self.seedream_seed if self.selected_model == MODEL_SEEDREAM else self.qwen_seed
    
    @property
    def high_quality(self) -> bool:
        """Получить настройку качества для текущей модели"""
        if self.selected_model == MODEL_SEEDREAM:
            return self.seedream_high_quality or self.seedream_4k_mode
        return self.qwen_high_quality
    
    def set_image_size(self, size: str) -> None:
        """Установить размер изображения для текущей модели"""
        if self.edit_mode_enabled:
            self.seedream_edit_image_size = size
        elif self.selected_model == MODEL_SEEDREAM:
            if size == "4K Квадрат":
                self.seedream_4k_mode = True
                self.seedream_image_size = "Портрет 9:16"  # Базовый размер
            else:
                self.seedream_4k_mode = False
                self.seedream_image_size = size
        else:
            self.qwen_image_size = size
    
    def set_safety_checker(self, enabled: bool) -> None:
        """Установить safety checker для текущей модели"""
        if self.selected_model == MODEL_SEEDREAM:
            self.seedream_enable_safety_checker = enabled
        else:
            self.qwen_enable_safety_checker = enabled
    
    def set_seed(self, seed: Union[int, None]) -> None:
        """Установить seed для текущей модели"""
        if self.selected_model == MODEL_SEEDREAM:
            self.seedream_seed = seed
        else:
            self.qwen_seed = seed
    
    def set_high_quality(self, enabled: bool) -> None:
        """Установить качество для текущей модели"""
        if self.selected_model == MODEL_SEEDREAM:
            if enabled and not self.seedream_4k_mode:
                # Если включаем HQ но не в 4K режиме
                self.seedream_high_quality = enabled
            elif not enabled:
                # Если выключаем HQ
                self.seedream_high_quality = False
                self.seedream_4k_mode = False
        else:
            self.qwen_high_quality = enabled
    
    def toggle_4k_mode(self) -> None:
        """Переключить 4K режим для SeDream"""
        if self.selected_model == MODEL_SEEDREAM:
            self.seedream_4k_mode = not self.seedream_4k_mode
            if self.seedream_4k_mode:
                self.seedream_high_quality = True  # 4K автоматически включает HQ

    
def _get_state(context: ContextTypes.DEFAULT_TYPE) -> SessionState:
    state = context.user_data.get("state")
    if isinstance(state, SessionState):
        return state
    state = SessionState()
    # По умолчанию при первом запуске выставляем режим редактирования выключенным и размер Auto 2K
    state.edit_mode_enabled = False
    state.seedream_edit_image_size = "Авто 2K"
    context.user_data["state"] = state
    return state


def _is_message_old(update: Update) -> bool:
    return False


def _is_group_chat(update: Update) -> bool:
    """Проверяет, является ли чат групповым."""
    if not update.effective_chat:
        return False
    return update.effective_chat.type in ['group', 'supergroup']


def _is_user_generating(user_id: int) -> bool:
    """Проверяет, генерирует ли пользователь изображение."""
    return ACTIVE_GENERATIONS.get(user_id, False)


def _set_user_generating(user_id: int, generating: bool) -> None:
    """Устанавливает статус генерации для пользователя."""
    if generating:
        ACTIVE_GENERATIONS[user_id] = True
    else:
        ACTIVE_GENERATIONS.pop(user_id, None)


def _load_stats() -> Dict[str, int]:
    """Загружает статистику из JSON файла."""
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load stats: %s", e)
    
    # Возвращаем дефолтную статистику
    return {
        "qwen_likes": 0, 
        "qwen_dislikes": 0
    }


def _save_stats(stats: Dict[str, int]) -> None:
    """Сохраняет статистику в JSON файл."""
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to save stats: %s", e)


def _update_stats(is_like: bool, model: str = "qwen") -> Dict[str, int]:
    """Обновляет статистику и возвращает новые значения."""
    stats = _load_stats()
    
    if is_like:
        stats[f"{model}_likes"] = stats.get(f"{model}_likes", 0) + 1
    else:
        stats[f"{model}_dislikes"] = stats.get(f"{model}_dislikes", 0) + 1
    
    _save_stats(stats)
    return stats


def _load_analytics() -> Dict[str, Any]:
    """Загружает аналитику бота из JSON файла."""
    try:
        if os.path.exists(BOT_ANALYTICS_FILE):
            with open(BOT_ANALYTICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load analytics: %s", e)
    
    # Возвращаем дефолтную аналитику
    return {
        "total_users": 0,
        "total_generations": 0,
        "qwen_generations": 0,
        "user_ids": []
    }


def _save_analytics(analytics: Dict[str, Any]) -> None:
    """Сохраняет аналитику бота в JSON файл."""
    try:
        with open(BOT_ANALYTICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to save analytics: %s", e)


def _update_user_analytics(user_id: int) -> None:
    """Обновляет аналитику пользователей (добавляет нового если его нет)."""
    analytics = _load_analytics()
    
    if user_id not in analytics["user_ids"]:
        analytics["user_ids"].append(user_id)
        analytics["total_users"] = len(analytics["user_ids"])
        _save_analytics(analytics)


def _increment_generation_count(model: str = "qwen") -> None:
    """Увеличивает счетчик сгенерированных изображений."""
    analytics = _load_analytics()
    analytics["total_generations"] += 1
    analytics[f"{model}_generations"] = analytics.get(f"{model}_generations", 0) + 1
    _save_analytics(analytics)


def _parse_flags_and_prompt(text: str, model: str) -> Tuple[str, Dict[str, Any]]:
    """Parse flags using the appropriate model's parser."""
    if model == MODEL_SEEDREAM:
        return SeDreamImageGenerator.parse_flags_and_prompt(text)
    # Default to Qwen
    return QwenImageGenerator.parse_flags_and_prompt(text)


def _parse_image_size(value: str) -> ImageSizeValue:
    """Parse image size using Qwen module."""
    return QwenImageGenerator.parse_image_size(value)


def _generate_images_via_fal(prompt: str, opts: Dict[str, Any], model: str = MODEL_QWEN) -> Dict[str, Any]:
    """Blocking call to FAL API through appropriate model module. Intended to run in a thread."""
    if model == MODEL_SEEDREAM:
        return generate_seedream_image(prompt, opts)
    else:
        return generate_qwen_image(prompt, opts)


def _format_settings_summary(state: SessionState) -> str:
    """Форматирует текущие настройки для показа пользователю."""
    # В режиме редактирования показываем специальные настройки
    if state.edit_mode_enabled:
        seed_display = str(state.seedream_seed) if state.seedream_seed is not None else "Авто"
        safety_display = "Вкл" if state.seedream_enable_safety_checker else "Выкл"
        return (
            f"<b>Размер:</b> <code>{state.seedream_edit_image_size}</code>\n"
            f"<b>Safety:</b> <code>{safety_display}</code>\n"
            f"<b>Seed:</b> <code>{seed_display}</code>\n"
            f"<b>Режим:</b> <code>Редактирование (SeDream Edit)</code>"
        )
    
    seed_display = str(state.seed) if state.seed is not None else "Авто"
    safety_display = "Вкл" if state.enable_safety_checker else "Выкл"
    
    # Разные настройки для разных моделей
    if state.selected_model == MODEL_SEEDREAM:
        if state.seedream_4k_mode:
            quality_display = "HQ 4K 🔥"
        elif state.high_quality:
            quality_display = "HQ файл"
        else:
            quality_display = "сжатое фото"
        
        # Для SeDream не показываем формат (всегда авто)
        return (
            f"<b>Размер:</b> <code>{state.image_size}</code>\n"
            f"<b>Safety:</b> <code>{safety_display}</code>\n"
            f"<b>Seed:</b> <code>{seed_display}</code>\n"
            f"<b>Качество:</b> <code>{quality_display}</code>"
        )
    else:
        # Для Qwen показываем формат
        quality_display = "HQ файл" if state.high_quality else "сжатое фото"
        return (
            f"<b>Размер:</b> <code>{state.image_size}</code>\n"
            f"<b>Safety:</b> <code>{safety_display}</code>\n"
            f"<b>Seed:</b> <code>{seed_display}</code>\n"
            f"<b>Качество:</b> <code>{quality_display}</code> | <b>Формат:</b> <code>PNG</code>"
        )


def _build_main_menu_markup(state: SessionState) -> InlineKeyboardMarkup:
    """Главное меню бота."""
    # Текст для кнопки переключения модели
    current_model_info = MODEL_INFO[state.selected_model]
    model_button_text = f"✦ {current_model_info['display_name']}"
    
    kb = []

    # Кнопка генерации/редактирования (на всю ширину)
    kb.append([
        InlineKeyboardButton(text=("🖼️ Редактировать изображение" if state.edit_mode_enabled else "🎨 Сгенерировать изображение"), callback_data="start_gen"),
    ])

    # Настройки и модель в одной строке (если режим редактирования выключен)
    if not state.edit_mode_enabled:
        kb.append([
            InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings"),
            InlineKeyboardButton(text=model_button_text, callback_data="model:switch"),
        ])
    else:
        # В режиме редактирования только настройки
        kb.append([
            InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings"),
        ])

    # Переключатель режима редактирования (на всю ширину)
    kb.append([
        InlineKeyboardButton(text=f"✏️ Режим редактирования: {'Вкл' if state.edit_mode_enabled else 'Выкл'}", callback_data="edit:toggle"),
    ])

    return InlineKeyboardMarkup(kb)


def _build_settings_markup(state: SessionState) -> InlineKeyboardMarkup:
    """Меню настроек."""
    safety_text = f"☁︎ Safety: {'вкл' if state.enable_safety_checker else 'выкл'}"
    
    kb = []
    
    if state.edit_mode_enabled:
        # Для редактирования доступны только авто размеры
        kb.append([
            InlineKeyboardButton(text="◄ Размер (Edit)", callback_data="editsz:prev"),
            InlineKeyboardButton(text="Размер (Edit) ►", callback_data="editsz:next"),
        ])
    else:
        # Размеры - скрываем если 4K режим
        hide_size = (state.selected_model == MODEL_SEEDREAM and state.seedream_4k_mode)
        
        if not hide_size:
            kb.append([
                InlineKeyboardButton(text="◄ Размер", callback_data="sz:prev"),
                InlineKeyboardButton(text="Размер ►", callback_data="sz:next"),
            ])
    
    # Safety и Quality в одной строке
    if state.edit_mode_enabled:
        quality_text = "Режим: Edit"
    elif state.selected_model == MODEL_SEEDREAM:
        # Для SeDream: сжатое → HQ → HQ 4K → сжатое
        if state.seedream_4k_mode:
            quality_text = "Качество: HQ 4K 🔥"
        elif state.high_quality:
            quality_text = "Качество: HQ файл"
        else:
            quality_text = "Качество: сжатое фото"
        
    else:
        # Для Qwen: сжатое → HQ → сжатое
        quality_text = f"Качество: {'HQ файл' if state.high_quality else 'сжатое фото'}"
    
    if state.edit_mode_enabled:
        kb.append([
            InlineKeyboardButton(text=safety_text, callback_data="sf:toggle"),
            InlineKeyboardButton(text=quality_text, callback_data="noop"),
        ])
    else:
        kb.append([
            InlineKeyboardButton(text=safety_text, callback_data="sf:toggle"),
            InlineKeyboardButton(text=quality_text, callback_data="hq:toggle"),
        ])

    # Seed настройки
    kb.append([
        InlineKeyboardButton(text="⛶ Новый сид", callback_data="seed:new"),
        InlineKeyboardButton(text="⛶ Сброс сида", callback_data="seed:auto"),
    ])
    
    # Назад
    kb.append([
        InlineKeyboardButton(text="◄ Назад", callback_data="menu"),
    ])
    
    return InlineKeyboardMarkup(kb)


def _build_confirmation_markup() -> InlineKeyboardMarkup:
    """Меню подтверждения генерации."""
    kb = [
        [
            InlineKeyboardButton(text="✅ Создать шедевр!", callback_data="confirm_gen"),
            InlineKeyboardButton(text="❌ Отмена", callback_data="cancel_gen"),
        ],
    ]
    return InlineKeyboardMarkup(kb)


def _build_feedback_markup() -> InlineKeyboardMarkup:
    """Кнопки обратной связи после генерации."""
    kb = [
        [
            InlineKeyboardButton(text="👍 Вау!", callback_data="feedback:wow"),
            InlineKeyboardButton(text="👎 Так себе", callback_data="feedback:meh"),
        ],
    ]
    return InlineKeyboardMarkup(kb)


def _build_new_generation_markup() -> InlineKeyboardMarkup:
    """Кнопка для начала новой генерации."""
    kb = [
        [
            InlineKeyboardButton(text="🎨 Создать что-то ещё", callback_data="menu"),
        ]
    ]
    return InlineKeyboardMarkup(kb)


async def _send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет главное меню."""
    state = _get_state(context)
    stats = _load_stats()
    
    # Получаем информацию о текущей модели
    current_model_info = MODEL_INFO[state.selected_model]
    model_name = current_model_info["name"]
    display_name = current_model_info["display_name"]
    description = current_model_info["description"]
    
    # Получаем статистику для текущей модели
    model_likes = stats.get(f"{state.selected_model}_likes", 0)
    model_dislikes = stats.get(f"{state.selected_model}_dislikes", 0)
    
    # Получаем информацию о лимите пользователя
    user_id = update.effective_user.id if update.effective_user else 0
    can_generate, remaining = _can_user_generate(user_id)
    
    text = (
        f"👋 <b>Приветствую, творец!</b> (v{BOT_VERSION})\n\n"
        f"Ваш инструмент сегодня — модель <b>{display_name}</b>.\n\n"
        f"<blockquote>{description}</blockquote>\n\n"
        f"<b>📊 Статистика модели {display_name}:</b>\n"
        f"👍 Лайки: <code>{model_likes}</code> | 👎 Дизы: <code>{model_dislikes}</code>\n\n"
        f"⏳ <b>Лимит:</b> <code>{remaining}/{USER_HOURLY_LIMIT}</code> час\n\n"
        "<b>Текущие настройки для генерации:</b>\n"
        + _format_settings_summary(state)
    )
    
    chat_id = update.effective_chat.id
    markup = _build_main_menu_markup(state)

    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=(text + ("\n\n✏️ <b>Режим редактирования включен.</b> Ответьте на фото вашим промптом или используйте команду /edit с изображением." if state.edit_mode_enabled else "")),
                reply_markup=markup,
                parse_mode="HTML",
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id, text=(text + ("\n\n✏️ <b>Режим редактирования включен.</b> Ответьте на фото вашим промптом или используйте команду /edit с изображением." if state.edit_mode_enabled else "")), reply_markup=markup, parse_mode="HTML"
            )
    except Forbidden:
        logger.debug("User %s blocked the bot", chat_id)
        return
    except (BadRequest, NetworkError, TelegramError) as e:
        logger.warning("⚠️ Menu send failed | %s", str(e))
        try:
            await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=markup, parse_mode="HTML"
            )
        except (Forbidden, TelegramError):
            logger.error("❌ Complete menu failure | Chat: %s", chat_id)
            return


async def _send_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет меню настроек."""
    state = _get_state(context)
    
    # Получаем информацию о текущей модели
    if state.edit_mode_enabled:
        model_display_name = "SeDream Edit"
    else:
        current_model_info = MODEL_INFO[state.selected_model]
        model_display_name = current_model_info["display_name"]
    
    text = (
        f"<b>⚙️ Палитра настроек</b>\n"
        f"<b>Модель:</b> {model_display_name}\n"
        + _format_settings_summary(state)
    )
    chat_id = update.effective_chat.id
    markup = _build_settings_markup(state)

    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=text,
                reply_markup=markup,
                parse_mode="HTML",
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=markup, parse_mode="HTML"
            )
    except Forbidden:
        logger.debug("User %s blocked the bot", chat_id)
        return
    except (BadRequest, NetworkError, TelegramError) as e:
        logger.warning("⚠️ Settings send failed | %s", str(e))
        try:
            await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=markup, parse_mode="HTML"
            )
        except (Forbidden, TelegramError):
            logger.error("❌ Complete settings failure | Chat: %s", chat_id)
            return


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /start - показывает главное меню."""
    # Игнорируем старые сообщения (отправленные до запуска бота)
    # Фильтр старых сообщений временно отключен
    
    # В группах не показываем меню, только приглашение в ЛС
    if _is_group_chat(update):
        await update.message.reply_text(
            f"👋 Привет из группы! Здесь я могу творить по команде /imagine. Для доступа ко всем настройкам и моделям, загляни ко мне в личные сообщения: {BOT_USERNAME}"
        )
        return
    
    await _send_main_menu(update, context)


async def imagine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    # Игнорируем старые сообщения (отправленные до запуска бота)
    # Фильтр старых сообщений временно отключен

    # Получаем состояние пользователя
    state = _get_state(context)

    # Если включен режим редактирования в ЛС — подсказка вместо генерации
    if not _is_group_chat(update) and _get_state(context).edit_mode_enabled:
        await update.message.reply_text(
            "✏️ Режим редактирования включен. Ответьте на фото вашим промптом или отправьте /edit вместе с картинкой. Чтобы вернуться к генерации — отключите режим в меню.")
        return

    # Обновляем аналитику пользователя
    if update.effective_user:
        _update_user_analytics(update.effective_user.id)

    # Проверяем, не генерирует ли уже пользователь
    user_id = update.effective_user.id if update.effective_user else 0
    # Лимит: не более 38 изображений за последний час
    can_generate, remaining = _can_user_generate(user_id)
    if not can_generate:
        await update.message.reply_text(
            "⏳ Лимит 38 изображений в час исчерпан. Попробуйте позже."
        )
        return
    if _is_user_generating(user_id):
        await update.message.reply_text(
            "⏳ Пожалуйста, подождите. Нейросеть уже трудится над вашим предыдущим запросом."
        )
        return

    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        await update.message.reply_text(
            "Ошибка: переменная окружения FAL_KEY не установлена.\n"
            "Установите ключ FAL и перезапустите бота."
        )
        return

    # Parse prompt and flags
    text = (" ".join(context.args)).strip() if context.args else ""
    if not text:
        await update.message.reply_text(
            "Без идеи нет шедевра. Укажите ваш замысел. Пример: /imagine кот-астронавт"
        )
        return

    prompt, parsed_opts = _parse_flags_and_prompt(text, state.selected_model)
    if not prompt:
        await update.message.reply_text("Ваша идея — это чистый холст. Пожалуйста, опишите её.")
        return

    # Получаем базовые настройки из сессии
    if state.selected_model == MODEL_SEEDREAM:
        api_size = SeDreamImageGenerator.convert_user_size_to_api(state.image_size)
        base_opts: Dict[str, Any] = {
            "image_size": api_size,
            "num_images": 1,
            "max_images": 1,
            "enable_safety_checker": state.enable_safety_checker,
            "sync_mode": False,
        }
    else:  # Qwen
        api_size = QwenImageGenerator.convert_user_size_to_api(state.image_size)
        base_opts: Dict[str, Any] = {
            "image_size": api_size,
            "num_inference_steps": 30,
            "guidance_scale": 2.5,
            "num_images": 1,
            "enable_safety_checker": state.enable_safety_checker,
            "output_format": "png",
            "negative_prompt": " ",
            "acceleration": "none",
        }
    if state.seed is not None:
        base_opts["seed"] = state.seed

    # Объединяем настройки: флаги из команды имеют приоритет
    opts = {**base_opts, **parsed_opts}

    # Log user generation request with emoji and clean format
    username = update.effective_user.username if update.effective_user else "unknown"
    chat_type = "Group" if _is_group_chat(update) else "Private"
    logger.info("🎯 User request | @%s | %s | Prompt: %.50s%s",
                username,
                chat_type,
                prompt,
                "..." if len(prompt) > 50 else "")

    try:
        # Устанавливаем статус генерации для пользователя
        _set_user_generating(user_id, True)
        
        await _send_chat_action(context, update, ChatAction.UPLOAD_PHOTO)
        
        # Специальное сообщение для 4K генерации
        if state.selected_model == MODEL_SEEDREAM and state.seedream_4k_mode:
            waiting_msg = await _send_waiting(update, context, "🔥 Создаю 4K шедевр... Это займет немного больше времени, но результат того стоит!")
        else:
            waiting_msg = await _send_waiting(update, context, "⏳ Нейросеть колдует... Обычно это занимает 10-30 секунд.")

        # Run blocking generation in a thread with selected model
        result: Dict[str, Any] = await asyncio.to_thread(_generate_images_via_fal, prompt, opts, state.selected_model)

        images: List[Dict[str, Any]] = result.get("images", []) if isinstance(result, dict) else []
        if not images:
            await waiting_msg.edit_text("🤔 Что-то пошло не так. Попробуйте переформулировать идею или использовать другие настройки.")
            _set_user_generating(user_id, False)
            return

        urls: List[str] = [img.get("url") for img in images if isinstance(img, dict) and img.get("url")]
        urls = [u for u in urls if u]

        if not urls:
            await waiting_msg.edit_text("😔 Не удалось загрузить результат. Пожалуйста, попробуйте еще раз через некоторое время.")
            _set_user_generating(user_id, False)
            return

        state = _get_state(context)
        # Получаем информацию о текущей модели
        current_model_info = MODEL_INFO[state.selected_model]
        model_display_name = current_model_info["display_name"]
        
        # Отмечаем потребление пользовательского лимита по числу отправленных изображений
        _mark_user_generated(user_id, images_count=len(urls))
        # Получаем обновленный лимит для отображения
        can_generate_after, remaining_after = _can_user_generate(user_id)
        
        caption = f"<code>Model:</code> {model_display_name}\n<code>Prompt:</code> {prompt}\n<code>Limit:</code> {remaining_after}/{USER_HOURLY_LIMIT} per hour"
        safe_caption, overflow = _split_caption(caption)

        if len(urls) == 1:
            if state.high_quality:
                await _reply_document(update, context, document=urls[0], caption=safe_caption, parse_mode="HTML")
            else:
                await _reply_photo(update, context, photo=urls[0], caption=safe_caption, parse_mode="HTML")
            # Без дополнительного сообщения при переполнении
        else:
            if state.high_quality:
                for i, url in enumerate(urls[:10]):
                    await _reply_document(update, context, document=url, caption=(safe_caption if i == 0 else None), parse_mode="HTML" if i == 0 else None)
            else:
                media = [InputMediaPhoto(media=u, caption=(safe_caption if i == 0 else None)) for i, u in enumerate(urls[:10])]
                await _reply_media_group(update, context, media=media)
            # Без дополнительного сообщения при переполнении

        # Увеличиваем счетчик сгенерированных изображений для текущей модели
        _increment_generation_count(state.selected_model)

        try:
            await waiting_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error("❌ Image generation failed | %s", str(e))
        await update.message.reply_text("💥 Упс, что-то пошло не так во время генерации. Попробуйте повторить запрос немного позже.")
    finally:
        # Всегда сбрасываем статус генерации
        _set_user_generating(user_id, False)


async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатий на кнопки."""
    if not update.callback_query:
        return
    query = update.callback_query
    state = _get_state(context)
    data = (query.data or "").strip()
    user_id = update.effective_user.id if update.effective_user else 0

    # Блокируем любые нажатия кнопок для пользователя, у которого сейчас идет генерация
    if _is_user_generating(user_id):
        try:
            await query.answer("⏳ Идет генерация. Дождитесь завершения.", show_alert=False)
        except Exception:
            pass
        return

    await query.answer()


    # Главное меню
    if data == "menu":
        # Сбрасываем состояние ожидания промпта
        state.awaiting_generation_prompt = False
        await _send_main_menu(update, context)
        return
    
    # Настройки
    if data == "settings":
        await _send_settings_menu(update, context)
        return
    
    # Переключение режима редактирования
    if data == "edit:toggle":
        state.edit_mode_enabled = not state.edit_mode_enabled
        # При включении режима редактирования принудительно выбираем модель SeDream и Авто 2K
        if state.edit_mode_enabled:
            state.selected_model = MODEL_SEEDREAM
            state.seedream_edit_image_size = "Авто 2K"  # Автоматически ставим Авто 2K
        await _send_main_menu(update, context)
        return
    
    # Начать генерацию (кнопка из главного меню)
    if data == "start_gen":
        # Создаем кнопку "Назад"
        back_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(text="◄ Назад", callback_data="menu")]
        ])

        if state.edit_mode_enabled:
            state.awaiting_generation_prompt = False
            await query.edit_message_text(
                text=(
                    "✏️ <b>Режим редактирования активен.</b>\n\n"
                    "1) Ответьте на фото вашим описанием правок.\n"
                    "2) Или пришлите фото с подписью /edit и описанием."
                ),
                parse_mode="HTML",
                reply_markup=back_keyboard
            )
        else:
            state.awaiting_generation_prompt = True
            await query.edit_message_text(
                text="✨ <b>Какую идею воплотим в жизнь?</b>\n\nОпишите ваш замысел. Чем детальнее, тем волшебнее будет результат! Например: <i>«Киберпанк-самурай под дождём неоновых огней»</i>",
                parse_mode="HTML",
                reply_markup=back_keyboard
            )
        return

    # Переключение модели
    if data == "model:switch":
        # Переключаем между моделями
        if state.edit_mode_enabled:
            # В режиме редактирования переключение модели недоступно
            await _send_main_menu(update, context)
            return
        if state.selected_model == MODEL_QWEN:
            state.selected_model = MODEL_SEEDREAM
        else:
            state.selected_model = MODEL_QWEN
        await _send_main_menu(update, context)
        return

    # Настройки размера
    if data.startswith("sz:"):
        if state.edit_mode_enabled:
            await _send_settings_menu(update, context)
            return
        current_model_info = MODEL_INFO[state.selected_model]
        allowed_sizes = current_model_info["allowed_sizes"]
        current_size = state.image_size
        
        try:
            idx = allowed_sizes.index(current_size)
        except ValueError:
            idx = 0
            
        if data.endswith("prev"):
            idx = (idx - 1) % len(allowed_sizes)
        else:
            idx = (idx + 1) % len(allowed_sizes)
            
        new_size = allowed_sizes[idx]
        state.set_image_size(new_size)
        await _send_settings_menu(update, context)
        return

    # Настройки размера (Edit)
    if data.startswith("editsz:"):
        allowed_sizes = SEEDREAM_EDIT_ALLOWED_SIZES
        current_size = state.seedream_edit_image_size
        try:
            idx = allowed_sizes.index(current_size)
        except ValueError:
            idx = 0
        if data.endswith("prev"):
            idx = (idx - 1) % len(allowed_sizes)
        else:
            idx = (idx + 1) % len(allowed_sizes)
        new_size = allowed_sizes[idx]
        state.set_image_size(new_size)
        await _send_settings_menu(update, context)
        return

    # Переключение safety
    if data == "sf:toggle":
        # В режиме редактирования всегда меняем настройку SeDream
        if state.edit_mode_enabled:
            state.seedream_enable_safety_checker = not state.seedream_enable_safety_checker
        else:
            state.set_safety_checker(not state.enable_safety_checker)
        await _send_settings_menu(update, context)
        return

    # Переключение качества (циклично)
    if data == "hq:toggle":
        if state.edit_mode_enabled:
            await _send_settings_menu(update, context)
            return
        if state.selected_model == MODEL_SEEDREAM:
            # Для SeDream: сжатое → HQ → HQ 4K → сжатое
            if state.seedream_4k_mode:
                # HQ 4K → сжатое
                state.seedream_4k_mode = False
                state.seedream_high_quality = False
            elif state.high_quality:
                # HQ → HQ 4K
                state.seedream_4k_mode = True
                state.seedream_high_quality = True
            else:
                # сжатое → HQ
                state.seedream_high_quality = True
        else:
            # Для Qwen: сжатое ↔ HQ
            state.set_high_quality(not state.high_quality)
        
        await _send_settings_menu(update, context)
        return

    # Заглушка для неактивных кнопок
    if data == "noop":
        # Просто игнорируем нажатие
        return

    # Управление seed
    if data == "seed:new":
        new_seed = random.randint(0, 2**31 - 1)
        # В режиме редактирования всегда меняем SeDream seed
        if state.edit_mode_enabled:
            state.seedream_seed = new_seed
        else:
            state.set_seed(new_seed)
        await _send_settings_menu(update, context)
        return

    if data == "seed:auto":
        # Проверяем, что seed не находится уже в авто режиме
        current_seed = state.seedream_seed if state.edit_mode_enabled else state.seed
        if current_seed is not None:
            if state.edit_mode_enabled:
                state.seedream_seed = None
            else:
                state.set_seed(None)
            await _send_settings_menu(update, context)
        # Если seed уже в авто режиме, просто игнорируем нажатие
        return

    # Обратная связь
    if data.startswith("feedback:"):
        # Убираем кнопки с исходного сообщения (с картинкой или текстом)
        await query.edit_message_reply_markup(reply_markup=None)

        # Обновляем статистику для текущей модели
        is_like = data.endswith("wow")
        # Определяем модель из контекста пользователя
        user_state = _get_state(context)
        current_model = user_state.selected_model
        updated_stats = _update_stats(is_like, current_model)
        
        if is_like:
            text = "🎉 Отлично! Рад, что вам понравилось. Вдохновение не ждёт, создадим что-нибудь ещё?"
            logger.info("👍 Positive feedback | %s Stats: 👍 %d | 👎 %d", 
                       current_model.upper(), 
                       updated_stats.get(f'{current_model}_likes', 0), 
                       updated_stats.get(f'{current_model}_dislikes', 0))
        else:  # feedback:meh
            text = "Понимаю, муза бывает капризной. Давайте попробуем другой подход?"
            logger.info("👎 Negative feedback | %s Stats: 👍 %d | 👎 %d", 
                       current_model.upper(),
                       updated_stats.get(f'{current_model}_likes', 0), 
                       updated_stats.get(f'{current_model}_dislikes', 0))

        # Проверяем, групповой ли это чат
        if _is_group_chat(update):
            # В группах отправляем сообщение с предложением перейти в ЛС
            group_text = f"{text}\n\n💬 Для доступа ко всем настройкам и моделям, загляни ко мне в личные сообщения: {BOT_USERNAME}"
            await _reply_text(update, context, group_text)
        else:
            # В ЛС отправляем с кнопкой "Создать что-то ещё"
            await _reply_text(update, context, text, reply_markup=_build_new_generation_markup())
        return

    # Заглушка удалена

    # Подтверждение генерации
    if data == "confirm_gen":
        # Удаляем сообщение с подтверждением
        try:
            await query.delete_message()
        except (Forbidden, TelegramError):
            pass
        
        # Сохраняем промпт из контекста
        prompt = context.user_data.get("pending_prompt", "")
        if prompt:
            await _generate_image(update, context, prompt)
        return
    
    # Отмена генерации
    if data == "cancel_gen":
        context.user_data.pop("pending_prompt", None)
        await _send_main_menu(update, context)
        return


async def _show_confirmation_screen(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    """Показывает экран подтверждения с настройками."""
    state = _get_state(context)
    
    # Получаем информацию о текущей модели
    current_model_info = MODEL_INFO[state.selected_model]
    model_display_name = current_model_info["display_name"]
    
    text = (
        f"✨ <b>Холст готов к работе!</b> ✨\n\n"
        f"<b>Модель:</b> <code>{model_display_name}</code>\n\n"
        f"<b>Ваша идея:</b>\n"
        f"<i>{html.escape(prompt)}</i>\n\n"
        f"<b>С такими настройками:</b>\n"
        + _format_settings_summary(state) +
        "\n\n"
        f"<i>Нажмите «❌ Отмена», чтобы вернуться в главное меню и изменить настройки или промпт.</i>"
    )
    
    # Сохраняем промпт для подтверждения
    context.user_data["pending_prompt"] = prompt
    
    markup = _build_confirmation_markup()
    chat_id = update.effective_chat.id
    
    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=text,
                reply_markup=markup,
                parse_mode="HTML",
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=markup, parse_mode="HTML"
            )
    except Forbidden:
        logger.debug("User %s blocked the bot", chat_id)
    except TelegramError as e:
        logger.error("❌ Confirmation screen failed | %s", str(e))


async def _generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    """Генерирует изображение по промпту."""
    state = _get_state(context)
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else 0

    # Обновляем аналитику пользователя (добавляем если не был добавлен)
    if update.effective_user:
        _update_user_analytics(update.effective_user.id)

    # Проверяем, не генерирует ли уже пользователь
    # Лимит: не более 38 изображений за последний час
    can_generate, remaining = _can_user_generate(user_id)
    if not can_generate:
        try:
            await context.bot.send_message(chat_id=chat_id, text="⏳ Лимит 38 изображений в час исчерпан. Попробуйте позже.")
        except Forbidden:
            logger.debug("User %s blocked the bot", chat_id)
        return
    if _is_user_generating(user_id):
        try:
            await context.bot.send_message(chat_id=chat_id, text="⏳ У вас уже идет генерация изображения. Дождитесь завершения.")
        except Forbidden:
            logger.debug("User %s blocked the bot", chat_id)
        return
        
    if not prompt.strip():
        try:
            await context.bot.send_message(chat_id=chat_id, text="Ваша идея — это чистый холст. Пожалуйста, опишите её.")
        except Forbidden:
            logger.warning("User %s blocked the bot", chat_id)
            return

    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        try:
            await context.bot.send_message(chat_id=chat_id, text="Ошибка: переменная окружения FAL_KEY не установлена.")
        except Forbidden:
            logger.warning("User %s blocked the bot", chat_id)
            return

    try:
        # Устанавливаем статус генерации для пользователя
        _set_user_generating(user_id, True)
        
        await _send_chat_action(context, update, ChatAction.UPLOAD_PHOTO)
        
        # Специальное сообщение для 4K генерации
        if state.selected_model == MODEL_SEEDREAM and state.seedream_4k_mode:
            loading_text = "🔥 Создаю 4K шедевр... Это займет немного больше времени, но результат того стоит!"
        else:
            CREATIVE_LOADING_MESSAGES = [
                "🎨 Смешиваю краски и пиксели...",
                "🧠 Нейросеть погрузилась в мир ваших фантазий...",
                "✨ Магия вот-вот произойдет на ваших глазах!",
                "🌌 Путешествую по вселенной идей в поисках шедевра...",
                "🤖 Мои роботы-художники уже взялись за кисти...",
                "⏳ Мгновение... и ваша идея станет реальностью!",
                "🔮 Заглядываю в будущее... там прекрасная картина!",
                "💡 Искра вдохновения зажглась! Начинаю творить...",
                "💻 Вычисляю идеальные пропорции красоты...",
                "💫 Собираю звездную пыль для вашего изображения...",
            ]
            loading_text = random.choice(CREATIVE_LOADING_MESSAGES)
        waiting_msg = await _send_waiting(update, context, loading_text)
    except Forbidden:
        logger.debug("User %s blocked the bot", chat_id)
        return
    except TelegramError as e:
        logger.error("❌ Generation start failed | %s", str(e))
        return

    # Мапим понятное название размера в API значение в зависимости от модели
    if state.selected_model == MODEL_SEEDREAM:
        api_size = SeDreamImageGenerator.convert_user_size_to_api(state.image_size)
    else:
        api_size = QwenImageGenerator.convert_user_size_to_api(state.image_size)
    
    # Настройки в зависимости от модели
    if state.selected_model == MODEL_SEEDREAM:
        opts: Dict[str, Any] = {
            "image_size": api_size,
            "num_images": 1,            # Всегда одно изображение
            "max_images": 1,           # Для SeDream
            "enable_safety_checker": state.enable_safety_checker,
            "sync_mode": False,        # Асинхронный режим
        }
    elif state.selected_model == MODEL_QWEN:
        # Настройки для Qwen
        opts: Dict[str, Any] = {
        "image_size": api_size,
        "num_inference_steps": 30,  # Фиксированное значение
        "guidance_scale": 2.5,      # Фиксированное значение
        "num_images": 1,            # Всегда одно изображение
        "enable_safety_checker": state.enable_safety_checker,
        "output_format": "png",     # Всегда PNG
        "negative_prompt": " ",     # Убираем негативный промпт
        "acceleration": "none",     # Фиксированное значение
    }
    else:
        opts = {}

    if state.seed is not None and opts is not None:
        opts["seed"] = state.seed

    # Log start of generation with user info
    username = update.effective_user.username if update.effective_user else "unknown"
    quality_mode = "HQ" if state.high_quality else "Standard"
    chat_type = "Group" if _is_group_chat(update) else "Private"
    logger.info("🚀 Starting generation | User: @%s | %s | Mode: %s | Prompt: %.50s%s",
                username, chat_type, quality_mode, prompt, "..." if len(prompt) > 50 else "")

    try:
        result: Dict[str, Any] = await asyncio.to_thread(_generate_images_via_fal, prompt, opts, state.selected_model)
        
        # Check for API errors
        error_message = result.get("error")
        if error_message:
            try:
                if "content_policy_violation" in str(error_message).lower() or "safety" in str(error_message).lower():
                    await waiting_msg.edit_text("⚠️ Сработал режим безопасности (цензура). Измените формулировку запроса и попробуйте снова.")
                else:
                    await waiting_msg.edit_text("😔 Ошибка на стороне сервиса генерации. Попробуйте немного позже.")
            except (Forbidden, TelegramError):
                pass
            _set_user_generating(user_id, False)
            return
        
        images: List[Dict[str, Any]] = result.get("images", []) if isinstance(result, dict) else []
        urls: List[str] = [img.get("url") for img in images if isinstance(img, dict) and img.get("url")]
        urls = [u for u in urls if u]
        if not urls:
            try:
                await waiting_msg.edit_text("🤔 Нейросеть не смогла создать изображение по вашему запросу. Попробуйте изменить идею или настройки.")
            except (Forbidden, TelegramError):
                pass
            _set_user_generating(user_id, False)
            return

        current_model_info = MODEL_INFO[state.selected_model]
        model_display_name = current_model_info["display_name"]
        
        # Отмечаем потребление пользовательского лимита
        _mark_user_generated(user_id, images_count=len(urls))
        # Получаем обновленный лимит для отображения
        can_generate_after, remaining_after = _can_user_generate(user_id)
        
        caption = f"<code>Model:</code> {model_display_name}\n<code>Prompt:</code> {prompt}\n<code>Limit:</code> {remaining_after}/{USER_HOURLY_LIMIT} per hour"
        safe_caption, overflow = _split_caption(caption)
        try:
            if len(urls) == 1:
                if state.high_quality:
                    # Отправляем как документ для высокого качества
                    await _reply_document(update, context, document=urls[0], caption=safe_caption, reply_markup=_build_feedback_markup(), parse_mode="HTML")
                else:
                    # Обычная отправка как фото (с сжатием)
                    await _reply_photo(update, context, photo=urls[0], caption=safe_caption, reply_markup=_build_feedback_markup(), parse_mode="HTML")
                if overflow:
                    await _reply_text(update, context, overflow)
            else:
                if state.high_quality:
                    # Для множественных изображений в HQ - отправляем как документы
                    for i, url in enumerate(urls[:10]):
                        await _reply_document(update, context, document=url, caption=(safe_caption if i == 0 else None), parse_mode="HTML" if i == 0 else None)
                    # Кнопки отправляем отдельным сообщением
                    await _reply_text(update, context, "Как вам результат?", reply_markup=_build_feedback_markup())
                else:
                    # Отправляем альбом фото (со сжатием)
                    media = [InputMediaPhoto(media=u, caption=(safe_caption if i == 0 else None)) for i, u in enumerate(urls[:10])]
                    await _reply_media_group(update, context, media=media)
                    
                    # Кнопки отправляем отдельным сообщением, так как альбомы не поддерживают markup
                    await _reply_text(update, context, "Как вам результат?", reply_markup=_build_feedback_markup())
                if overflow:
                    await _reply_text(update, context, overflow)

            # Увеличиваем счетчик сгенерированных изображений для текущей модели
            _increment_generation_count(state.selected_model)

        except Forbidden:
            logger.debug("User %s blocked the bot during result sending", chat_id)
            return
        except TelegramError as e:
            logger.error("❌ Failed to send images | %s", str(e))
            try:
                await _reply_text(update, context, "✅ Изображение готово, но у меня возникли проблемы с его отправкой. Попробуйте снова.")
            except (Forbidden, TelegramError):
                pass
            return
        
        try:
            await waiting_msg.delete()
        except (Forbidden, TelegramError):
            pass
    except Exception as e:
        logger.error("💥 Generation failed | %s", str(e))
        try:
            await waiting_msg.edit_text("💥 Упс, магия не сработала... Попробуйте повторить запрос немного позже.")
        except (Forbidden, TelegramError):
            try:
                await context.bot.send_message(chat_id=chat_id, text="💥 Упс, магия не сработала... Попробуйте повторить запрос немного позже.")
            except (Forbidden, TelegramError):
                pass
    finally:
        # Всегда сбрасываем статус генерации
        _set_user_generating(user_id, False)


async def handle_photo_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает фотографии с подписями."""
    is_group = _is_group_chat(update)
    state = _get_state(context)
    caption = (update.message.caption or "") if update.message else ""
    
    # В группах: если подпись начинается с /edit - редактируем
    if is_group and caption.strip().startswith("/edit"):
        prompt = caption.strip()[5:].strip()  # Убираем "/edit" и пробелы
        if not prompt:
            await update.message.reply_text("Опишите, что изменить после команды /edit")
            return
        image_urls = await _extract_image_urls(update, context)
        if not image_urls:
            await update.message.reply_text("Не удалось получить файл изображения.")
            return
        await perform_edit(update, context, prompt, image_urls)
        return
    
    # В ЛС: при включенном режиме редактирования можно отправить фото с подписью (без /edit)
    if not is_group and state.edit_mode_enabled:
        prompt = caption.strip()
        if not prompt:
            await update.message.reply_text("Пришлите описание правок в подписи к фото или ответьте текстом на фото.")
            return
        image_urls = await _extract_image_urls(update, context)
        if not image_urls:
            await update.message.reply_text("Не удалось получить файл изображения.")
            return
        await perform_edit(update, context, prompt, image_urls)
        return


async def stat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /stat - показывает статистику бота."""
    # Игнорируем старые сообщения (отправленные до запуска бота)
    # Фильтр старых сообщений временно отключен
    
    analytics = _load_analytics()
    feedback_stats = _load_stats()
    
    # Форматируем статистику для обеих моделей
    stat_text = (
        f"📊 <b>Статистика творчества v{BOT_VERSION}</b>\n\n"
        f"👥 <b>Всего пользователей:</b> <code>{analytics['total_users']}</code>\n\n"
        f"🎨 <b>Генерации изображений:</b>\n"
        f"• Всего: <code>{analytics['total_generations']}</code>\n"
        f"• Qwen-Image: <code>{analytics.get('qwen_generations', 0)}</code>\n"
        f"• SeDream 4.0: <code>{analytics.get('seedream_generations', 0)}</code>\n\n"
        f"<b>📈 Обратная связь Qwen-Image:</b>\n"
        f"• Лайки 👍: <code>{feedback_stats.get('qwen_likes', 0)}</code>\n"
        f"• Дизлайки 👎: <code>{feedback_stats.get('qwen_dislikes', 0)}</code>\n\n"
        f"<b>📈 Обратная связь SeDream 4.0:</b>\n"
        f"• Лайки 👍: <code>{feedback_stats.get('seedream_likes', 0)}</code>\n"
        f"• Дизлайки 👎: <code>{feedback_stats.get('seedream_dislikes', 0)}</code>\n\n"
        f"<i>Данные обновляются в реальном времени ✨</i>"
    )
    
    try:
        await update.message.reply_text(stat_text, parse_mode="HTML")
    except Forbidden:
        logger.debug("User %s blocked the bot", update.effective_chat.id)
    except TelegramError as e:
        logger.error("❌ Stat command failed | %s", str(e))


async def edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /edit — редактирует изображения по промпту и входному фото.
    В группах работает всегда. В ЛС — только когда включен режим редактирования.
    Поддерживает как ответ на фото, так и команду в подписи к фото.
    """
    if update.message is None:
        return

    is_group = _is_group_chat(update)
    state = _get_state(context)

    if not is_group and not state.edit_mode_enabled:
        await update.message.reply_text("Режим редактирования выключен. Включите его в главном меню.")
        return

    prompt = (" ".join(context.args)).strip() if context.args else ""
    
    # Если команда без аргументов, но есть caption - используем caption как промпт
    if not prompt and update.message.caption:
        # Убираем /edit из caption если он там есть
        caption_text = update.message.caption.strip()
        if caption_text.startswith("/edit"):
            prompt = caption_text[5:].strip()  # Убираем "/edit" и пробелы
        else:
            prompt = caption_text
    
    if not prompt:
        # Если команда пришла без описания
        await update.message.reply_text("Опишите, что изменить. Пример: /edit сделай фон неоновым и добавь кота")
        return

    image_urls = await _extract_image_urls(update, context)
    if not image_urls:
        await update.message.reply_text("Прикрепите изображение или ответьте командой на сообщение с изображением.")
        return

    await perform_edit(update, context, prompt, image_urls)


async def perform_edit(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, image_urls: List[str]) -> None:
    """Выполняет редактирование изображений через SeDream Edit."""
    user_id = update.effective_user.id if update.effective_user else 0
    # Лимиты
    can_generate, _ = _can_user_generate(user_id)
    if not can_generate:
        await _reply_text(update, context, "⏳ Лимит 38 изображений в час исчерпан. Попробуйте позже.")
        return
    if _is_user_generating(user_id):
        await _reply_text(update, context, "⏳ Пожалуйста, дождитесь завершения текущей обработки.")
        return

    state = _get_state(context)
    # Размер по умолчанию авто 2К
    api_size = convert_user_edit_size_to_api(state.seedream_edit_image_size)
    opts: Dict[str, Any] = {
        "image_size": api_size,
        "num_images": 1,
        "max_images": 1,
        "enable_safety_checker": state.seedream_enable_safety_checker,  # Edit всегда использует SeDream
        "sync_mode": False,
    }
    if state.seedream_seed is not None:  # Edit всегда использует SeDream seed
        opts["seed"] = state.seedream_seed

    _set_user_generating(user_id, True)
    try:
        await _send_chat_action(context, update, ChatAction.UPLOAD_PHOTO)
        waiting = await _send_waiting(update, context, "✏️ Редактирую изображение…")
        result: Dict[str, Any] = await asyncio.to_thread(generate_seedream_edit, prompt, image_urls, opts)
        images: List[Dict[str, Any]] = result.get("images", []) if isinstance(result, dict) else []
        urls: List[str] = [img.get("url") for img in images if isinstance(img, dict) and img.get("url")]
        urls = [u for u in urls if u]
        if not urls:
            err = result.get("error") if isinstance(result, dict) else None
            try:
                await waiting.edit_text("Не удалось отредактировать изображение." + (f" Ошибка: {err}" if err else ""))
            except Exception:
                pass
            return

        _mark_user_generated(user_id, images_count=len(urls))
        can_after, remaining_after = _can_user_generate(user_id)
        caption = f"<code>Mode:</code> Edit (SeDream 4)\n<code>Prompt:</code> {prompt}\n<code>Limit:</code> {remaining_after}/{USER_HOURLY_LIMIT} per hour"
        safe_caption, overflow = _split_caption(caption)
        
        # В ЛС показываем кнопки оценки, в группах — нет
        is_group = _is_group_chat(update)
        feedback_markup = None if is_group else _build_feedback_markup()
        
        # При "Авто 4K" отправляем без сжатия (как документ)
        is_4k_mode = state.seedream_edit_image_size == "Авто 4K"
        
        if len(urls) == 1:
            if is_4k_mode:
                await _reply_document(update, context, document=urls[0], caption=safe_caption, reply_markup=feedback_markup, parse_mode="HTML")
            else:
                await _reply_photo(update, context, photo=urls[0], caption=safe_caption, reply_markup=feedback_markup, parse_mode="HTML")
        else:
            if is_4k_mode:
                # Для 4K отправляем как документы без сжатия
                for i, url in enumerate(urls[:10]):
                    await _reply_document(update, context, document=url, caption=(safe_caption if i == 0 else None), parse_mode="HTML" if i == 0 else None)
                # Кнопки отдельным сообщением в ЛС
                if not is_group:
                    await _reply_text(update, context, "Как вам результат?", reply_markup=_build_feedback_markup())
            else:
                media = [InputMediaPhoto(media=u, caption=(safe_caption if i == 0 else None)) for i, u in enumerate(urls[:10])]
                await _reply_media_group(update, context, media=media)
                # Для множественных изображений в ЛС отправляем кнопки отдельным сообщением
                if not is_group:
                    await _reply_text(update, context, "Как вам результат?", reply_markup=_build_feedback_markup())
        try:
            await waiting.delete()
        except Exception:
            pass
    except Exception as e:
        logger.error("Edit failed: %s", e)
        await _reply_text(update, context, "💥 Ошибка при редактировании. Попробуйте позже.")
    finally:
        _set_user_generating(user_id, False)


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений."""
    if not update.message or not update.message.text:
        return
    
    # Игнорируем старые сообщения (отправленные до запуска бота)
    # Фильтр старых сообщений временно отключен
    
    is_group = _is_group_chat(update)
    
    state = _get_state(context)
    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    
    # Если включен режим редактирования в ЛС и это ответ на фото — редактируем
    if not is_group:
        if state.edit_mode_enabled and text and getattr(update.message, "reply_to_message", None):
            replied = update.message.reply_to_message
            has_image = bool(getattr(replied, "photo", None) or (getattr(replied, "document", None) and getattr(replied.document, "mime_type", "").startswith("image/")))
            if has_image:
                # Запускаем редактирование
                image_urls = await _extract_image_urls(update, context)
                if image_urls:
                    await perform_edit(update, context, text, image_urls)
                    return
                else:
                    await update.message.reply_text("Не нашёл изображение для редактирования. Пришлите фото заново.")
                    return

    # Если ожидаем промпт для генерации
    if state.awaiting_generation_prompt and not is_group:
        state.awaiting_generation_prompt = False
        if text:
            # Показываем экран подтверждения
            await _show_confirmation_screen(update, context, text)
        else:
            try:
                await update.message.reply_text("Кажется, вы забыли описать свою идею. Попробуйте снова.")
            except Forbidden:
                logger.debug("User %s blocked the bot", chat_id)
            return


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler to log errors and prevent crashes."""
    
    # Only try to send error message if we have an update with a chat
    if isinstance(update, Update) and update.effective_chat:
        chat_id = update.effective_chat.id
        error_msg = str(context.error) if context.error else "Unknown error"
        
        if isinstance(context.error, Forbidden):
            logger.debug("User %s blocked the bot", chat_id)
            return
        elif isinstance(context.error, (BadRequest, NetworkError)):
            logger.warning("⚠️ Telegram API issue | %s", error_msg)
            return
        else:
            # Log unexpected errors with context
            logger.error("💥 Unexpected error | Chat: %s | %s", chat_id, error_msg)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⚙️ Произошла непредвиденная ошибка. Я уже сообщил о ней, попробуйте повторить ваш запрос позже."
                )
            except (Forbidden, TelegramError):
                logger.debug("Failed to send error message to user %s", chat_id)
    else:
        # Log errors without chat context
        error_msg = str(context.error) if context.error else "Unknown error"
        logger.error("💥 Global error | %s", error_msg)


def main() -> None:
    global BOT_START_TIME
    
    # Устанавливаем время запуска бота
    BOT_START_TIME = time.time()
    
    # Load .env if present
    try:
        load_dotenv()
    except Exception:
        pass

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN не задан. Установите переменную окружения и запустите снова.")
        raise SystemExit(1)

    builder = ApplicationBuilder().token(bot_token)
    try:
        # Включаем параллельную обработку апдейтов, чтобы один долгий хендлер не блокировал других
        application = builder.concurrent_updates(True).build()
    except Exception:
        # На случай, если метод недоступен в текущей версии библиотеки
        application = builder.build()
    logger.info("📝 Registering handlers...")
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", start))
    application.add_handler(CommandHandler("imagine", imagine))
    application.add_handler(CommandHandler("edit", edit_command))
    application.add_handler(CommandHandler("stat", stat))  # Команда статистики
    application.add_handler(CallbackQueryHandler(handle_buttons))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))
    # Фото для режима редактирования: в ЛС всегда при включенном режиме, в группах только с /edit в подписи
    application.add_handler(MessageHandler(filters.PHOTO | (filters.Document.IMAGE), handle_photo_input))
    logger.info("✅ All handlers registered")
    
    # Add global error handler
    application.add_error_handler(error_handler)

    logger.info("🤖 Qwen Image Bot v%s starting...", BOT_VERSION)
    logger.info("🕐 Startup time: %s", time.strftime("%H:%M:%S", time.localtime(BOT_START_TIME)))
    logger.info("✅ Bot ready! Waiting for messages...")
    
    try:
        # Игнорируем все накопившиеся обновления до запуска (официальный способ)
        application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error("💥 Bot crashed | %s", str(e))
        raise


if __name__ == "__main__":
    main()



