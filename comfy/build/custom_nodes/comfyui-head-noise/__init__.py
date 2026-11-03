from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

from .nodes import NODES


class HeadNoiseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> HeadNoiseExtension:
    return HeadNoiseExtension()
