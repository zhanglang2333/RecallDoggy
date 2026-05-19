import json
import importlib.util
import os
import sys
import subprocess
import shutil

PLUGINS_DIR = "/root/RecallDoggy/plugins"


class PluginLoader:
    def __init__(self):
        self.plugins = {}
        self._registered = {}  # name -> {"routes": [route_obj], "tools": [tool_name]}
        os.makedirs(PLUGINS_DIR, exist_ok=True)

    # ── scan / load ──────────────────────────────────

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

    # ── MCP tool helpers ─────────────────────────────

    def _get_mcp_tool_names(self, mcp_server):
        if mcp_server is None:
            return set()
        if hasattr(mcp_server, "_tool_manager"):
            mgr = mcp_server._tool_manager
            if hasattr(mgr, "_tools"):
                return set(mgr._tools.keys())
            if hasattr(mgr, "tools"):
                return set(mgr.tools.keys())
        if hasattr(mcp_server, "_tools"):
            return set(mcp_server._tools.keys())
        if hasattr(mcp_server, "tools"):
            return set(mcp_server.tools.keys())
        return set()

    def _remove_mcp_tool(self, mcp_server, tool_name):
        if mcp_server is None:
            return
        if hasattr(mcp_server, "_tool_manager"):
            mgr = mcp_server._tool_manager
            if hasattr(mgr, "_tools"):
                mgr._tools.pop(tool_name, None)
                return
            if hasattr(mgr, "tools"):
                mgr.tools.pop(tool_name, None)
                return
        if hasattr(mcp_server, "_tools"):
            mcp_server._tools.pop(tool_name, None)
        elif hasattr(mcp_server, "tools"):
            mcp_server.tools.pop(tool_name, None)

    # ── load / unload / reload ───────────────────────

    def _load_one(self, name, app, mcp_server=None):
        p = self.plugins[name]
        entry = p["meta"].get("entry", "main.py")
        entry_path = os.path.join(p["dir"], entry)
        if not os.path.exists(entry_path):
            return

        # snapshot before
        routes_before = set(id(r) for r in app.routes)
        tools_before = self._get_mcp_tool_names(mcp_server)

        # clean old module cache
        mod_name = f"plugin_{name}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        spec = importlib.util.spec_from_file_location(mod_name, entry_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        p["module"] = mod

        if hasattr(mod, "register"):
            mod.register(app, mcp_server)

        # snapshot after — record diff
        new_routes = [r for r in app.routes if id(r) not in routes_before]
        new_tools = self._get_mcp_tool_names(mcp_server) - tools_before

        self._registered[name] = {
            "routes": new_routes,
            "tools": list(new_tools),
        }

    def _unload_one(self, name, app, mcp_server=None):
        reg = self._registered.pop(name, None)
        if reg:
            for route in reg["routes"]:
                try:
                    app.routes.remove(route)
                except ValueError:
                    pass
            for tool_name in reg["tools"]:
                self._remove_mcp_tool(mcp_server, tool_name)

        mod_name = f"plugin_{name}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        if name in self.plugins:
            self.plugins[name]["module"] = None

    def reload_plugin(self, name, app, mcp_server=None):
        """hot reload: unload -> re-read meta -> load"""
        if name not in self.plugins:
            raise NameError(f"插件 '{name}' 不存在")

        self._unload_one(name, app, mcp_server)

        # re-read plugin.json
        p = self.plugins[name]
        meta_path = os.path.join(p["dir"], "plugin.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                p["meta"] = json.load(f)
            p["meta"].setdefault("enabled", True)

        if p["meta"].get("enabled", True):
            self._load_one(name, app, mcp_server)
            return {"status": "reloaded", "name": name}
        return {"status": "disabled_skip", "name": name}

    # ── install / uninstall / toggle ─────────────────

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
        self._registered.pop(name, None)

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
                "settings_schema": p["meta"].get("settings_schema", []),
                "settings_url": p["meta"].get("settings_url"),
            }
            for p in self.plugins.values()
        ]

    def get_plugin_settings(self, name):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
        cfg_path = os.path.join(base, name, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        for p in self.plugins.values():
            if p["meta"]["name"] == name:
                return {s["key"]: s.get("default", "") for s in p["meta"].get("settings_schema", [])}
        return {}

    def save_plugin_settings(self, name, data):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
        cfg_path = os.path.join(base, name, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
