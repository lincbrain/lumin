import numpy as np
import neuroglancer as ng
from typing import Optional, List


class Viewer:
    def __init__(self, image_url: str, model: str):
        self.num_actions = 0
        self.model = model
        self.viewer = ng.Viewer()

        # basic, universal viewer functions
        with self.viewer.txn() as s:
            s.layers["image"] = ng.ImageLayer(source=image_url)
            # s.layers["segmentation"] = ng.SegmentationLayer(source=segmentation_url)
            s.show_slices = False
            s.concurrent_downloads = 256
            s.gpu_memory_limit = 2 * 1024 * 1024 * 1024
            s.layout = "3d"

    def segment_volume(self, action):
        """segment a volume using the specified model"""
        if self.model == "cellpose":
            print(f"hello")
            segmentation = None
        else:
            raise NotImplementedError

        # add segmentation layer to viewer
        with self.viewer.txn() as s:
            s.layers["segmentation"] = ng.SegmentationLayer(
                source=ng.LocalVolume(data=segmentation),
            )

    def _register_callback(self):
        """register a segmentation callback function
        with neuroglancer's API"""
        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.viewer["keyt"] = "segment_volume"

    def _move_voxel(self, voxel_coordinates: Optional[List] = None):
        """move viewer position to specified coordinates"""
        if coords is not None:
            with self.viewer.txn() as s:
                s.voxel_coordinates = voxel_coordinates

    def _bind_action(self, key):
        """bind key-mappings to neuroglancer UI for backend model computation"""
        self.num_actions += 1

        with self.viewer.config_state.txn() as st:
            st.status_messages["hello"] = f"Got action {num_actions}: mouse position = "

    def _hide_segmentation(self, hide_seg: Optional[bool] = False):
        """toggle viewing of segmentation layer"""
        if hide_seg:
            with self.viewer.txn() as s:
                s.layers["segmentation"].visible = False

    def display_viewer(self):
        """display the viewer"""
        print(f"viewer: {self.viewer}")


if __name__ == "__main__":
    v = Viewer(
        image_url="https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/",
        model="abc",
    )
    v.display_viewer()
