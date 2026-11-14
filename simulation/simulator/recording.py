"""Offscreen PNG snapshots and a streamed GIF, without a video codec dependency."""

from pathlib import Path
import mujoco as mj
from PIL import Image, ImageDraw, GifImagePlugin


class Recorder:
    def __init__(self, model, directory, enabled=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.renderer = mj.Renderer(model, height=480, width=640) if enabled else None
        self.stream = (self.directory / "simulation.gif").open("wb") if enabled else None
        self.frames = 0
        self.next_time = 0.
        self.label = "Initial setup"
        self.camera = mj.MjvCamera()
        self.camera.lookat[:] = [.40, 0., .25]
        self.camera.distance = 1.65
        self.camera.azimuth = 135
        self.camera.elevation = -40
        self.palette = Image.new("P", (1,1))
        colors = [v for r in range(6) for g in range(6) for b in range(6) for v in (r*51,g*51,b*51)]
        self.palette.putpalette(colors + [0]*(768-len(colors)))

    def frame(self, data, force=False, snapshot=None):
        if self.renderer is None:
            return
        if not force and data.time < self.next_time:
            return
        self.next_time = data.time + 1.2
        self.renderer.update_scene(data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0,0,640,28), fill=(20,24,30))
        draw.text((10,8), f"{self.label}   |   simulation {data.time:.1f} s", fill="white")
        if snapshot:
            frame.save(self.directory / f"{snapshot}.png")
        indexed = frame.quantize(palette=self.palette, dither=Image.Dither.NONE)
        if not self.frames:
            header, _ = GifImagePlugin.getheader(indexed, info={"loop":0})
            for chunk in header:
                self.stream.write(chunk)
        for chunk in GifImagePlugin.getdata(indexed, duration=1500 if snapshot else 75):
            self.stream.write(chunk)
        self.frames += 1

    def close(self):
        if self.stream:
            self.stream.write(b";")
            self.stream.close()
            self.renderer.close()

    def grid_snapshot(self, data, name):
        if self.renderer is None:
            return
        self.renderer.update_scene(data,camera="grid_view")
        frame=Image.fromarray(self.renderer.render())
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,640,28),fill=(20,24,30))
        draw.text((10,8),self.label,fill="white")
        frame.save(self.directory / f"{name}_grid.png")
