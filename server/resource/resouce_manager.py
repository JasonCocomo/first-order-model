from collections import OrderedDict
from .resource import VideoResource
from .ppd_generator import PPDGenerator
import torch
import os


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> int:
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class ResourceManager():

    def __init__(self, max_count,
                 ppd_generator: PPDGenerator):
        self.cache = LRUCache(max_count)
        self.ppd_generator = ppd_generator

    def get_cache_data(self, material_file_path, ppd_file_path):
        if material_file_path not in self.cache:
            self.add_cache_data(material_file_path, ppd_file_path)
        cache_data = self.cache[material_file_path]
        return cache_data

    def add_cache_data(self, material_file_path, ppd_file_path):
        resource = VideoResource(material_file_path)
        resource.load_material()
        if not os.path.exists(ppd_file_path):
            ppd_file = self.ppd_generator.generate(resource, ppd_file_path)
        else:
            ppd_file = torch.load(ppd_file_path)
        self.cache.put((resource, ppd_file))
