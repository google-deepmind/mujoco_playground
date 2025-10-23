import pymeshlab
import glob
ms = pymeshlab.MeshSet()

for filename in glob.glob("*.stl"):
    print(f"Processing {filename}")
    ms.load_new_mesh(filename)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_decimation_quadric_edge_collapse()
    ms.meshing_decimation_quadric_edge_collapse()
    ms.meshing_decimation_quadric_edge_collapse()
    ms.meshing_decimation_quadric_edge_collapse()
    ms.save_current_mesh(filename)