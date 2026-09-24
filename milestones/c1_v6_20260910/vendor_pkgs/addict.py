# -*- coding: utf-8 -*-
"""addict 最小替代（离线 vendor，供 modelscope 使用）。
addict.Dict：字典属性访问。modelscope 只用到这一功能。"""
__all__ = ["Dict"]


class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                super().__setitem__(k, Dict(v))
            elif isinstance(v, list):
                super().__setitem__(k, [Dict(x) if isinstance(x, dict) else x
                                        for x in v])

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = Dict(value)
        self[name] = value
