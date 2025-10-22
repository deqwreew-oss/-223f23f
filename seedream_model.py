"""
SeDream v4 Image Generation Model Module
Модуль для работы с моделью Bytedance SeDream v4 через FAL API
"""

import os
import re
import logging
from typing import Any, Dict, List, Union
from dataclasses import dataclass

import fal_client


# Настройка логгера для модуля
logger = logging.getLogger("seedream-model")


# Типы для размеров изображений
ImageSizeValue = Union[str, Dict[str, int]]

# Размеры для SeDream v4 (поддерживает до 4K)
SEEDREAM_SIZE_MAP = {
    "Квадрат 2K": {"width": 2048, "height": 2048},
    "Квадрат HD": "square_hd",
    "Портрет 9:16": "portrait_16_9", 
    "Альбом 16:9": "landscape_16_9",
    "Альбом 4:3": "landscape_4_3",
    "4K Квадрат": {"width": 4096, "height": 4096},  # Специальный 4K режим
}

SEEDREAM_ALLOWED_SIZES: List[str] = list(SEEDREAM_SIZE_MAP.keys())

# Размеры для режима редактирования (только авто 2K/4K)
SEEDREAM_EDIT_SIZE_MAP = {
    "Авто 2K": "auto_2K",
    "Авто 4K": "auto_4K",
}
SEEDREAM_EDIT_ALLOWED_SIZES: List[str] = list(SEEDREAM_EDIT_SIZE_MAP.keys())


@dataclass
class SeDreamGenerationOptions:
    """Настройки генерации для модели SeDream v4"""
    image_size: Union[str, Dict[str, int]] = "portrait_16_9"
    num_images: int = 1
    max_images: int = 1
    enable_safety_checker: bool = False  # По умолчанию выключен для SeDream
    seed: Union[int, None] = None
    sync_mode: bool = False


@dataclass
class SeDreamGenerationResult:
    """Результат генерации изображения SeDream v4"""
    success: bool
    images: List[Dict[str, Any]]
    seed: Union[int, None] = None
    error: Union[str, None] = None
    
    @property
    def image_urls(self) -> List[str]:
        """Возвращает список URL изображений"""
        if not self.success or not self.images:
            return []
        
        urls = []
        for img in self.images:
            if isinstance(img, dict) and img.get("url"):
                urls.append(img["url"])
        return urls


class SeDreamImageGenerator:
    """Класс для работы с моделью SeDream v4 через FAL API"""
    
    MODEL_ENDPOINT = "fal-ai/bytedance/seedream/v4/text-to-image"
    MODEL_NAME = "SeDream v4"
    MODEL_DISPLAY_NAME = "SeDream 4.0"
    
    def __init__(self, fal_api_key: str = None):
        """
        Инициализация генератора
        
        Args:
            fal_api_key: API ключ для FAL, если None - берется из переменной окружения
        """
        self.fal_api_key = fal_api_key or os.getenv("FAL_KEY")
        if not self.fal_api_key:
            raise ValueError("FAL_KEY не установлен. Укажите API ключ или установите переменную окружения.")
    
    @staticmethod
    def parse_image_size(value: str) -> ImageSizeValue:
        """
        Парсит размер изображения из строки для SeDream v4
        
        Args:
            value: строка с размером (например "4096x4096" или "portrait_16_9")
            
        Returns:
            Размер в формате API или словарь с width/height
        """
        v = value.strip().lower()
        
        # Проверяем формат widthxheight (например, 4096x4096)
        match = re.match(r"^(\d{3,5})x(\d{3,5})$", v)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            
            # SeDream поддерживает от 1024 до 4096
            if 1024 <= width <= 4096 and 1024 <= height <= 4096:
                return {"width": width, "height": height}
            else:
                # Ограничиваем размеры если выходят за пределы
                width = max(1024, min(4096, width))
                height = max(1024, min(4096, height))
                return {"width": width, "height": height}

        # Проверяем предустановленные размеры API
        allowed_api_sizes = {
            "square_hd", "square", "portrait_4_3", "portrait_16_9",
            "landscape_4_3", "landscape_16_9"
        }
        
        if v in allowed_api_sizes:
            return v
            
        # По умолчанию возвращаем портретный формат для SeDream
        return "portrait_16_9"
    
    @staticmethod
    def parse_flags_and_prompt(text: str) -> tuple[str, Dict[str, Any]]:
        """
        Парсит флаги и промпт из текста команды для SeDream v4
        
        Args:
            text: текст команды с возможными флагами
            
        Returns:
            Кортеж (промпт, словарь_опций)
        """
        # Разбиваем текст с учетом кавычек
        try:
            import shlex
            tokens = shlex.split(text)
        except Exception:
            tokens = text.split()

        options: Dict[str, Any] = {}
        prompt_tokens: List[str] = []

        for token in tokens:
            if token.startswith("--") and "=" in token:
                key, value = token[2:].split("=", 1)
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")

                if key in {"size", "image_size"}:
                    options["image_size"] = SeDreamImageGenerator.parse_image_size(value)
                elif key in {"seed"}:
                    try:
                        options["seed"] = int(value)
                    except ValueError:
                        pass
                elif key in {"images", "num_images"}:
                    try:
                        # SeDream поддерживает больше изображений
                        options["num_images"] = max(1, min(8, int(value)))
                    except ValueError:
                        pass
                elif key in {"max_images"}:
                    try:
                        options["max_images"] = max(1, min(8, int(value)))
                    except ValueError:
                        pass
                elif key in {"safety", "enable_safety_checker"}:
                    options["enable_safety_checker"] = value.lower() in {"1", "true", "yes", "on"}
                elif key in {"sync", "sync_mode"}:
                    options["sync_mode"] = value.lower() in {"1", "true", "yes", "on"}
            else:
                prompt_tokens.append(token)

        prompt = " ".join(prompt_tokens).strip()
        return prompt, options
    
    def generate_image(self, prompt: str, options: SeDreamGenerationOptions = None) -> SeDreamGenerationResult:
        """
        Генерирует изображение по промпту (синхронная функция)
        
        Args:
            prompt: текстовое описание изображения
            options: настройки генерации
            
        Returns:
            Результат генерации
        """
        if not prompt.strip():
            return SeDreamGenerationResult(success=False, images=[], error="Промпт не может быть пустым")
        
        if options is None:
            options = SeDreamGenerationOptions()
        
        # Подготавливаем аргументы для API
        arguments = {
            "prompt": prompt,
            "image_size": options.image_size,
            "num_images": options.num_images,
            "max_images": options.max_images,
            "enable_safety_checker": options.enable_safety_checker,
            "sync_mode": options.sync_mode,
        }
        
        if options.seed is not None:
            arguments["seed"] = options.seed

        # Логируем начало генерации
        is_4k = isinstance(options.image_size, dict) and options.image_size.get("width", 0) >= 4096
        size_info = f"4K ({options.image_size['width']}x{options.image_size['height']})" if is_4k else str(options.image_size)
        
        logger.info("🎨 Starting SeDream v4 generation | Size: %s | Images: %s | Safety: %s", 
                    size_info,
                    options.num_images,
                    "ON" if options.enable_safety_checker else "OFF")

        def on_queue_update(update: Any) -> None:
            # Подавляем внутренние логи FAL API
            pass

        try:
            result: Dict[str, Any] = fal_client.subscribe(
                self.MODEL_ENDPOINT,
                arguments=arguments,
                with_logs=False,
                on_queue_update=on_queue_update,
            )
            
            if not isinstance(result, dict):
                error_msg = f"Invalid response format: {type(result)}"
                logger.error("❌ SeDream API error | %s", error_msg)
                return SeDreamGenerationResult(success=False, images=[], error=error_msg)
            
            images = result.get("images", [])
            if not images:
                return SeDreamGenerationResult(success=False, images=[], error="no_images")
            
            # Логируем успешную генерацию
            images_count = len(images)
            returned_seed = result.get("seed")
            logger.info("✅ SeDream v4 generation completed | Images: %d | Seed: %s", 
                       images_count, returned_seed)
            
            return SeDreamGenerationResult(
                success=True, 
                images=images, 
                seed=returned_seed
            )
            
        except Exception as e:
            # Улучшенная обработка ошибок FAL API
            error_msg = SeDreamImageGenerator._parse_error_message(e)
            logger.error("❌ SeDream API failed | %s", error_msg)
            return SeDreamGenerationResult(success=False, images=[], error=error_msg)
    
    def generate_from_command_text(self, command_text: str) -> SeDreamGenerationResult:
        """
        Генерирует изображение из текста команды с флагами
        
        Args:
            command_text: текст команды (например "/imagine кот --size=4096x4096")
            
        Returns:
            Результат генерации
        """
        prompt, parsed_options = self.parse_flags_and_prompt(command_text)
        
        if not prompt:
            return SeDreamGenerationResult(success=False, images=[], error="Промпт пуст")
        
        # Создаем настройки генерации из парсенных опций
        options = SeDreamGenerationOptions()
        
        # Применяем парсенные опции
        if "image_size" in parsed_options:
            options.image_size = parsed_options["image_size"]
        if "num_images" in parsed_options:
            options.num_images = parsed_options["num_images"]
        if "max_images" in parsed_options:
            options.max_images = parsed_options["max_images"]
        if "enable_safety_checker" in parsed_options:
            options.enable_safety_checker = parsed_options["enable_safety_checker"]
        if "seed" in parsed_options:
            options.seed = parsed_options["seed"]
        if "sync_mode" in parsed_options:
            options.sync_mode = parsed_options["sync_mode"]
        
        return self.generate_image(prompt, options)

    @staticmethod
    def convert_user_size_to_api(user_size: str) -> Union[str, Dict[str, int]]:
        """
        Конвертирует пользовательский размер в API формат
        
        Args:
            user_size: размер в понятном формате (например "4K Квадрат")
            
        Returns:
            Размер в API формате
        """
        return SEEDREAM_SIZE_MAP.get(user_size, "portrait_16_9")
    
    @staticmethod
    def is_4k_size(size_value: Union[str, Dict[str, int]]) -> bool:
        """
        Проверяет, является ли размер 4K
        
        Args:
            size_value: значение размера
            
        Returns:
            True если это 4K размер
        """
        if isinstance(size_value, dict):
            width = size_value.get("width", 0)
            height = size_value.get("height", 0)
            return width >= 4096 or height >= 4096
        return False
    
    @staticmethod
    def _parse_error_message(error: Exception) -> str:
        """
        Парсит сообщение об ошибке от FAL API
        
        Args:
            error: исключение от FAL API
            
        Returns:
            Читаемое сообщение об ошибке
        """
        error_str = str(error)
        
        # Проверяем на content policy violation
        if "content_policy_violation" in error_str.lower():
            return "content_policy_violation"
        
        # Пытаемся распарсить JSON ошибку если это список
        try:
            # Ищем JSON в строке ошибки
            import json
            import re
            
            # Ищем JSON массив в строке
            json_match = re.search(r'\[{.*}\]', error_str)
            if json_match:
                error_list = json.loads(json_match.group())
                if isinstance(error_list, list) and len(error_list) > 0:
                    first_error = error_list[0]
                    if isinstance(first_error, dict):
                        # Возвращаем тип ошибки если есть
                        error_type = first_error.get("type", "")
                        error_msg = first_error.get("msg", "")
                        
                        if error_type == "content_policy_violation":
                            return "content_policy_violation"
                        elif error_msg:
                            return error_msg
                        elif error_type:
                            return error_type
            
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Если не удалось распарсить JSON, возвращаем исходную ошибку
            pass
        
        return error_str


# Утилитарные функции для обратной совместимости
def create_seedream_generator() -> SeDreamImageGenerator:
    """Создает экземпляр генератора SeDream v4"""
    return SeDreamImageGenerator()


def generate_seedream_image(prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Генерирует изображение через SeDream v4 (для обратной совместимости)
    
    Args:
        prompt: промпт для генерации
        options: словарь с настройками
        
    Returns:
        Словарь с результатом в старом формате
    """
    try:
        generator = create_seedream_generator()
        
        # Конвертируем старый формат опций в новый
        seedream_options = SeDreamGenerationOptions()
        if options:
            if "image_size" in options:
                seedream_options.image_size = options["image_size"]
            if "num_images" in options:
                seedream_options.num_images = options["num_images"]
            if "max_images" in options:
                seedream_options.max_images = options["max_images"]
            if "enable_safety_checker" in options:
                seedream_options.enable_safety_checker = options["enable_safety_checker"]
            if "seed" in options:
                seedream_options.seed = options["seed"]
            if "sync_mode" in options:
                seedream_options.sync_mode = options["sync_mode"]
        
        result = generator.generate_image(prompt, seedream_options)
        
        # Возвращаем в старом формате для совместимости
        if result.success:
            response = {"images": result.images}
            if result.seed is not None:
                response["seed"] = result.seed
            return response
        else:
            return {"images": [], "error": result.error}
            
    except Exception as e:
        # Используем улучшенную обработку ошибок
        error_msg = SeDreamImageGenerator._parse_error_message(e)
        return {"images": [], "error": error_msg}

@dataclass
class SeDreamEditOptions:
    """Настройки редактирования для модели SeDream v4 Edit"""
    image_size: Union[str, Dict[str, int]] = "auto_2K"
    num_images: int = 1
    max_images: int = 1
    enable_safety_checker: bool = True
    seed: Union[int, None] = None
    sync_mode: bool = False


class SeDreamEditGenerator:
    """Класс для редактирования изображений SeDream v4 Edit через FAL API"""

    MODEL_ENDPOINT = "fal-ai/bytedance/seedream/v4/edit"

    def __init__(self, fal_api_key: str = None):
        self.fal_api_key = fal_api_key or os.getenv("FAL_KEY")
        if not self.fal_api_key:
            raise ValueError("FAL_KEY не установлен. Укажите API ключ или установите переменную окружения.")

    def edit_image(self, prompt: str, image_urls: List[str], options: SeDreamEditOptions = None) -> SeDreamGenerationResult:
        if not prompt.strip():
            return SeDreamGenerationResult(success=False, images=[], error="Промпт не может быть пустым")
        if not image_urls:
            return SeDreamGenerationResult(success=False, images=[], error="Не передано изображение для редактирования")

        options = options or SeDreamEditOptions()

        arguments: Dict[str, Any] = {
            "prompt": prompt,
            "image_urls": image_urls[:10],
            "image_size": options.image_size,
            "num_images": options.num_images,
            "max_images": options.max_images,
            "enable_safety_checker": options.enable_safety_checker,
            "sync_mode": options.sync_mode,
        }
        if options.seed is not None:
            arguments["seed"] = options.seed

        try:
            result: Dict[str, Any] = fal_client.subscribe(
                self.MODEL_ENDPOINT,
                arguments=arguments,
                with_logs=False,
            )

            if not isinstance(result, dict):
                error_msg = f"Invalid response format: {type(result)}"
                logger.error("❌ SeDream Edit API error | %s", error_msg)
                return SeDreamGenerationResult(success=False, images=[], error=error_msg)

            images = result.get("images", [])
            if not images:
                return SeDreamGenerationResult(success=False, images=[], error="no_images")

            returned_seed = result.get("seed")
            return SeDreamGenerationResult(success=True, images=images, seed=returned_seed)

        except Exception as e:
            error_msg = SeDreamImageGenerator._parse_error_message(e)
            logger.error("❌ SeDream Edit API failed | %s", error_msg)
            return SeDreamGenerationResult(success=False, images=[], error=error_msg)


def convert_user_edit_size_to_api(user_size: str) -> Union[str, Dict[str, int]]:
    """Преобразует человеко-понятное имя размера редактирования в API значение."""
    return SEEDREAM_EDIT_SIZE_MAP.get(user_size, "auto_2K")


def generate_seedream_edit(prompt: str, image_urls: List[str], options: Dict[str, Any] = None) -> Dict[str, Any]:
    """Редактирует изображение через SeDream v4 Edit и возвращает результат в старом формате."""
    try:
        generator = SeDreamEditGenerator()

        edit_options = SeDreamEditOptions()
        if options:
            if "image_size" in options:
                edit_options.image_size = options["image_size"]
            if "num_images" in options:
                edit_options.num_images = options["num_images"]
            if "max_images" in options:
                edit_options.max_images = options["max_images"]
            if "enable_safety_checker" in options:
                edit_options.enable_safety_checker = options["enable_safety_checker"]
            if "seed" in options:
                edit_options.seed = options["seed"]
            if "sync_mode" in options:
                edit_options.sync_mode = options["sync_mode"]

        result = generator.edit_image(prompt, image_urls, edit_options)
        if result.success:
            response = {"images": result.images}
            if result.seed is not None:
                response["seed"] = result.seed
            return response
        else:
            return {"images": [], "error": result.error}
    except Exception as e:
        error_msg = SeDreamImageGenerator._parse_error_message(e)
        return {"images": [], "error": error_msg}
