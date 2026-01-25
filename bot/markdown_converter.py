import html
import re

class TelegramMarkdownConverter:
    """
    Класс для конвертации Markdown от LLM в валидный HTML для Telegram
    и безопасного разбиения сообщений.
    """

    @staticmethod
    def to_html(text: str) -> str:
        """
        Преобразует Markdown-подобный текст в Telegram HTML.
        """
        # 1. Экранируем весь текст (защита от инъекций и < > в коде)
        text = html.escape(text, quote=False)

        # 2. Разбиваем текст на части: код и не код
        # Паттерн ищет ```язык ... ```
        parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
        
        final_parts = []
        
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                # --- ЭТО БЛОК КОДА ---
                # Убираем кавычки
                content = part[3:-3].strip()
                
                # Попытка определить язык в первой строке
                first_line_end = content.find('\n')
                if first_line_end != -1 and first_line_end < 20:
                    lang_candidate = content[:first_line_end].strip()
                    # Проверяем валидность названия языка (буквы, цифры, +, #)
                    if re.match(r'^[a-zA-Z0-9+#]+$', lang_candidate):
                        content = content[first_line_end+1:]
                
                # Оборачиваем в pre
                final_parts.append(f"<pre>{content}</pre>")
            else:
                # --- ЭТО ОБЫЧНЫЙ ТЕКСТ ---
                # Жирный: **text** -> <b>text</b>
                part = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', part)
                
                # Инлайн код: `text` -> <code>text</code>
                part = re.sub(r'`([^`]+)`', r'<code>\1</code>', part)
                
                # Ссылки: [text](url) -> <a href="url">text</a>
                # Исправленный regex
                part = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', part)
                
                final_parts.append(part)
                
        return "".join(final_parts)

    @staticmethod
    def split_safe(text: str, limit: int = 3000) -> list[str]:
        """
        Разбивает текст на куски по параграфам, чтобы не превысить лимит Telegram.
        """
        chunks = []
        current_chunk = ""
        
        # Разбиваем по двойным переносам (абзацам)
        paragraphs = text.split('\n\n')
        
        for p in paragraphs:
            # Если добавление параграфа превысит лимит
            if len(current_chunk) + len(p) + 2 > limit:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Если сам параграф больше лимита (например, огромный код)
                if len(p) > limit:
                    for i in range(0, len(p), limit):
                        chunks.append(p[i:i+limit])
                else:
                    current_chunk = p
            else:
                if current_chunk:
                    current_chunk += "\n\n" + p
                else:
                    current_chunk = p
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks