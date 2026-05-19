"""Skill loader — 读写 SKILL.md"""

from pathlib import Path
from typing import Optional
from datetime import datetime


class SkillManager:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.global_skill = ""
        self.plugin_skills = {}

    def load_global(self):
        path = self.base_dir / "SKILL.md"
        if path.exists():
            self.global_skill = path.read_text(encoding="utf-8")
        return self.global_skill

    def load_plugin_skill(self, plugin_name):
        path = self.base_dir / "plugins" / plugin_name / "SKILL.md"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            self.plugin_skills[plugin_name] = content
            return content
        return None

    def list_all(self):
        result = []
        global_path = self.base_dir / "SKILL.md"
        if global_path.exists():
            result.append({"name": "global", "path": "SKILL.md"})
        plugins_dir = self.base_dir / "plugins"
        if plugins_dir.exists():
            for skill_file in plugins_dir.glob("*/SKILL.md"):
                plugin_name = skill_file.parent.name
                result.append({"name": plugin_name, "path": f"plugins/{plugin_name}/SKILL.md"})
        return result

    def append_lesson(self, lesson):
        path = self.base_dir / "SKILL.md"
        content = path.read_text(encoding="utf-8") if path.exists() else "# Skills\n\n## 经验教训\n"
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        entry = f"- {date_prefix}: {lesson}"
        marker = "## 经验教训"
        if marker in content:
            parts = content.split(marker, 1)
            after = parts[1]
            next_section = after.find("\n## ", 1)
            if next_section == -1:
                content = content.rstrip() + f"\n{entry}\n"
            else:
                before_next = after[:next_section]
                rest = after[next_section:]
                content = parts[0] + marker + before_next.rstrip() + f"\n{entry}\n" + rest
        else:
            content = content.rstrip() + f"\n\n## 经验教训\n{entry}\n"
        path.write_text(content, encoding="utf-8")
        self.global_skill = content
