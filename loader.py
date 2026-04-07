import json
import importlib.util
import os
import subprocess
import shutil

PLUGINS_DIR = "/root/plugins"


class PluginLoader:
    def __init__(self):
        self.plugins = {}
        os.makedirs(PLUGINS_DIR, exist_ok=True)

    def scan(self):
        self.plugins = {}
        for name in os.listdir(PLUGINS_DIR):
            plugin_dir = os.path.join(PLUGINS_DIR, name)
            meta_path = os.path.join(plugin_dir, "plugin.json")
            if os.path.isdir(plugin_dir) and os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta.setdefault("enabled", True)
                self.plugins[meta["name"]] = {
                    "meta": meta,
                    "dir": plugin_dir,
                    "module": None,
                }

    def load_all(self, app, mcp_server=None):
        self.scan()
        for name, p in self.plugins.items():
            if p["meta"].get("enabled", True):
                self._load_one(name, app, mcp_server)

    def _load_one(self, name, app, mcp_server=None):
        p = self.plugins[name]
        entry = p["meta"].get("entry", "main.py")
        entry_path = os.path.join(p["dir"], entry)
        if not os.path.exists(entry_path):
            return
        spec = importlib.util.spec_from_file_location(f"plugin_{name}", entry_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        p["module"] = mod
        if hasattr(mod, "register"):
            mod.register(app, mcp_server)

    def install(self, url: str) -> dict:
        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
        target = os.path.join(PLUGINS_DIR, repo_name)
        if os.path.exists(target):
            raise ValueError(f"'{repo_name}' 已存在")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, target],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(f"clone失败: {result.stderr}")
        meta_path = os.path.join(target, "plugin.json")
        if not os.path.exists(meta_path):
            shutil.rmtree(target)
            raise FileNotFoundError("没有plugin.json，不是合法插件")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def uninstall(self, name: str):
        if name not in self.plugins:
            raise NameError(f"插件 '{name}' 不存在")
        shutil.rmtree(self.plugins[name]["dir"])
        del self.plugins[name]

    def toggle(self, name: str) -> bool:
        if name not in self.plugins:
            raise NameError(f"插件 '{name}' 不存在")
        meta = self.plugins[name]["meta"]
        meta["enabled"] = not meta.get("enabled", True)
        meta_path = os.path.join(self.plugins[name]["dir"], "plugin.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta["enabled"]

    def list_plugins(self) -> list:
        self.scan()
        return [
            {
                "name": p["meta"]["name"],
                "version": p["meta"].get("version", "?"),
                "description": p["meta"].get("description", ""),
                "enabled": p["meta"].get("enabled", True),
            }
            for p in self.plugins.values()
        ]