import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from functools import wraps
from pathlib import Path
from typing import Union, List
import os
import json
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import openfoamparser_mai as Ofpp
import pyvista

def save_json(filename, data, save_path) -> None:
    """Cохраняет json"""
    file_path = save_path / Path(filename)
    with open(file_path, 'w', encoding="utf8") as f:
        json.dump(data, f)


def save_json_in_chunks(filename, data, save_path, chunk_size=1000):
    full_path = os.path.join(save_path, filename)
    with open(full_path, 'w') as file:
        file.write('[')
        for i, item in enumerate(data):
            json_str = json.dumps(item)
            file.write(json_str)
            if i < len(data) - 1:
                file.write(',\n')
            if i % chunk_size == 0 and i != 0:
                file.flush()  # Flush data to disk periodically
        file.write(']')


# The wrapper function for multiprocessing
def save_json_in_chunks_wrapper(args):
    save_json_in_chunks(*args)


def json_streaming_writer(filepath, data_func, data_args):
    """Write JSON data to a file using a generator to minimize memory usage."""
    data_gen = data_func(*data_args)
    try:
        with open(filepath, 'w') as file:
            print(f"writing {filepath}")
            file.write('[')
            for i, item in enumerate(data_gen):
                if i != 0:  # Add a comma before all but the first item
                    file.write(',')
                json.dump(item, file)
            file.write(']')
        print(f"Finished writing {filepath}")
    except Exception as e:
        print(f"Failed to write {filepath}: {str(e)}") 


def create_nodes_gen(mesh_bin):
    """Generator for nodes."""
    for point in mesh_bin.points:
        yield {
            'X': point[0],
            'Y': point[1],
            'Z': point[2]
        }


def create_faces_gen(mesh_bin):
    """Generator for faces."""
    for face in mesh_bin.faces:
        yield list(face)


def create_elements_gen(mesh_bin, p, u, c):
    """Generator for elements."""
    for i, cell in enumerate(mesh_bin.cell_faces):
        yield {
            'Faces': cell,
            'Pressure': p[i],
            'Velocity': {
                'X': u[i][0],
                'Y': u[i][1],
                'Z': u[i][2]
            },
            'VelocityModule': np.linalg.norm(u[i]),
            'Position': {
                'X': c[i][0],
                'Y': c[i][1],
                'Z': c[i][2]
            }
        }


def create_surfaces_gen(surfaces):
    """Generator for surfaces."""
    for surface in surfaces:
        yield surface


def _face_center_position(points: list, mesh: Ofpp.FoamMesh) -> list:
    vertecis = [mesh.points[p] for p in points]
    vertecis = np.array(vertecis)
    return list(vertecis.mean(axis=0))
def pressure_field_on_surface(solver_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 surface_name: str = 'Surface') -> None:
    """Поле давлений на поверхности тела:
    'Nodes' - List[x: float, y: float, z:float], 
    'Faces' - List [List[int]], 
    'Elements' - List [Dict{Faces: List[int],
                            Pressure: float,
                            Velocity: List[float],
                            VelocityModule: float,
                            Position: List[float]}
                            ], 
    'Surfaces' - List[
                    Tuple[Surface_name: str, 
                    List[Dict{ParentElementID: int,
                              ParentFaceId: int,
                              Position: List[float]}]
                    ]

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с расчетом.
        p (np.ndarray): Поле давления.
        surface_name (str): Имя для поверхности.
    """
    
    # Step 0: parse mesh and scale vertices
    mesh_bin = Ofpp.FoamMesh(solver_path )

    # Step I: compute TFemFace_Surface
    domain_names = ["motorBike_0".encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))

        body_faces = []
        for f, b in f_b_set:
            try:
                position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'CentrePosition': {'X': position[0], 'Y': position[1], 'Z': position[2]},
                    'PressureValue': p[b]
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,
                'Item2': body_faces}) 
        
        return surfaces
# generate all data and save as a .npy file for Sphere_stationary
def gain_all_point_data_Sphere_stationary():
    rootdir = 'F:\work\AI Challenge\case_2_field_prediction-main\Sphere_stationary'
    list = os.listdir(rootdir)
    data = []
    for name in list:
        PATH_TO_CASE = os.path.join(rootdir, name)
        if name == '0.3M':
            angle = 0
            v = 102.09000405669215
            END_TIME = '150'
        elif name == '0.5M':
            angle = 0
            v = 170.15
            END_TIME = '150'
        elif name == '0.7M':
            angle = 0
            v = 238.20999999999998
            END_TIME = '150'
        elif name == '0.9M':
            angle = 0
            v = 306.27000000000004
            END_TIME = '150'
        elif name == '1.1M':
            angle = 1
            v = 374.33000000000004
            END_TIME = '40'
        elif name == '1.3M':
            angle = 1
            v = 442.39000000000004
            END_TIME = '40'
        elif name == '1.5M':
            angle = 1
            v = 510.45000000000005
            END_TIME = '40'
        elif name == '1.7M':
            angle = 1
            v = 578.51
            END_TIME = '40'
        elif name == '1.9M':
            angle = 1
            v = 646.5699999999999
            END_TIME = '40'
        elif name == '2.0M':
            angle = 1
            v = 680.6
            END_TIME = '40'
        base_path = Path(PATH_TO_CASE)
        print(PATH_TO_CASE)
        time_path = base_path / Path(END_TIME)
        p_path = time_path / Path('p')
        p = Ofpp.parse_internal_field(p_path)
        surface = pressure_field_on_surface(base_path, p)
        for s in surface[0]['Item2']:
            tmp = np.array([angle, v, s['CentrePosition']['X'], s['CentrePosition']['Y'], s['CentrePosition']['Z'], s['PressureValue']])
            data.append(tmp)
    np.save("data.npy",data)
# generate all data and save as a .npy file for crm_example
def gain_all_point_data_crm_example():
    rootdir = 'F://work//AI Challenge//test1//crm_example'
    data = []
    PATH_TO_CASE = rootdir
    END_TIME = '200'
    angle = 0
    v = 170.15
    base_path = Path(PATH_TO_CASE)
    print(PATH_TO_CASE)
    time_path = base_path / Path(END_TIME)
    p_path = time_path / Path('p')
    print(p_path)
    p = Ofpp.parse_internal_field(p_path)
    surface = pressure_field_on_surface(base_path, p)
    for s in surface[0]['Item2']:
        tmp = np.array([angle, v, s['CentrePosition']['X'], s['CentrePosition']['Y'], s['CentrePosition']['Z'], s['PressureValue']])
        data.append(tmp)
    np.save("data_crm_example.npy",data)
# generate all data and save as a .npy file for agard
def gain_all_point_data_agard():
    rootdir = 'F://work//AI Challenge//test1//agard'
    list = os.listdir(rootdir)
    data = []
    for name in list:
        PATH_TO_CASE = os.path.join(rootdir, name)
        if name == 'agard_0.6M_05deg':
            angle = 17.795459554216844
            v = 203.4030334563726
            END_TIME = '150'
        elif name == 'agard_0.7V_15Deg':
            angle = 61.65328473387146
            v = 230.09319108031895
            END_TIME = '150'
        elif name == 'agard150.0':
            angle = 0
            v = 150
            END_TIME = '150'
        elif name == 'agard183.33333333333334':
            angle = 0
            v = 183.33333333333334
            END_TIME = '150'
        elif name == 'agard216.66666666666669':
            angle = 0
            v = 216.66666666666669
            END_TIME = '150'
        elif name == 'agard250.0':
            angle = 0
            v = 250.0
            END_TIME = '150'
        elif name == 'agard280.0':
            angle = 0
            v = 280.0
            END_TIME = '150'
        elif name == 'agard303.3333333333333':
            angle = 0
            v = 303.3333333333333
            END_TIME = '150'
        elif name == 'agard326.6666666666667':
            angle = 0
            v = 326.6666666666667
            END_TIME = '150'
        elif name == 'agard350.0':
            angle = 1
            v = 350.0
            END_TIME = '150'
        base_path = Path(PATH_TO_CASE)
        print(PATH_TO_CASE)
        time_path = base_path / Path(END_TIME)
        p_path = time_path / Path('p')
        p = Ofpp.parse_internal_field(p_path)
        surface = pressure_field_on_surface(base_path, p)
        for s in surface[0]['Item2']:
            tmp = np.array([angle, v, s['CentrePosition']['X'], s['CentrePosition']['Y'], s['CentrePosition']['Z'], s['PressureValue']])
            data.append(tmp)
    np.save("data_agard.npy",data)
# generate all data and save as a .npy file for princess_luna
def gain_all_point_data_luna():
    rootdir = 'princess_luna'
    list = os.listdir(rootdir)
    data = []
    for name in list:
        PATH_TO_CASE = os.path.join(rootdir, name)
        if name == '0.3M':
            angle = 0
            v = 102.09000405669215
            END_TIME = '150'
        elif name == '0.5M':
            angle = 0
            v = 170.15
            END_TIME = '150'
        elif name == '0.7M':
            angle = 0
            v = 238.20999999999998
            END_TIME = '150'
        elif name == '0.9M':
            angle = 0
            v = 306.27000000000004
            END_TIME = '150'
        elif name == '1.1М':
            angle = 1
            v = 374.33000000000004
            END_TIME = '40'
        elif name == '1.3M':
            angle = 1
            v = 442.39000000000004
            END_TIME = '40'
        elif name == '1.5M':
            angle = 1
            v = 510.45000000000005
            END_TIME = '40'
        elif name == '1.7M':
            angle = 1
            v = 578.51
            END_TIME = '40'
        elif name == '1.9M':
            angle = 1
            v = 646.5699999999999
            END_TIME = '40'
        elif name == '2.0M':
            angle = 1
            v = 680.6
            END_TIME = '40'
        base_path = Path(PATH_TO_CASE)
        time_path = base_path / Path(END_TIME)
        p_path = time_path / Path('p')
        print(p_path)
        p = Ofpp.parse_internal_field(p_path)
        surface = pressure_field_on_surface(base_path, p)
        for s in surface[0]['Item2']:
            tmp = np.array([angle, v, s['CentrePosition']['X'], s['CentrePosition']['Y'], s['CentrePosition']['Z'], s['PressureValue']])
            data.append(tmp)
    np.save("data_luna.npy",data)
# generate all data and save as a .npy file for sphere_transiet
def gain_all_point_data_sphere_transiet():
    rootdir = 'sphere_transiet'
    list = os.listdir(rootdir)
    data = []
    for name in list:
        list1 = os.listdir(Path(rootdir) / Path(name))
        if name == 'rhoPimple0.5':
            angle = 0
            v = 160
        elif name == 'rhoPimple0.7':
            angle = 0
            v = 240
        elif name == 'rhoPimple0.8':
            angle = 0
            v = 300
        PATH_TO_CASE = os.path.join(rootdir, name)
        for name1 in list1:
            if name1 != '0.orig' and name != 'constant' and name != 'system' and name != 'foam.foam':
                END_TIME = name1
                base_path = Path(PATH_TO_CASE)
                time_path = base_path / Path(END_TIME)
                p_path = time_path / Path('p')
                print(p_path)
                p = Ofpp.parse_internal_field(p_path)
                surface = pressure_field_on_surface(base_path, p)
                for s in surface[0]['Item2']:
                    tmp = np.array([angle, v, s['CentrePosition']['X'], s['CentrePosition']['Y'], s['CentrePosition']['Z'], s['PressureValue']])
                    data.append(tmp)
    np.save("data_sphere_transiet.npy",data)

def Norm(X, max, min):
    if max != min:
        return (X - min) / (max - min)
    else:
        return (X - min) / (0.001)



class Mydataset(Dataset):
    def __init__(self, path):
        data = np.load(path)  # you can choose "data_crm_example.npy" / "data_Sphere_stationary.npy" / "data_agard.npy" / "data_luna.npy"
        angle = []
        v = []
        x = []
        y = []
        z = []
        p = []
        for i in range(len(data)):
            angle.append(data[i][0])
            v.append(data[i][1])
            x.append(data[i][2])
            y.append(data[i][3])
            z.append(data[i][4])
            p.append(data[i][5])
        self.angle = np.array(angle)
        self.v = np.array(v)
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.p = np.array(p)
        self.path = path

    def __len__(self):
        return len(self.p)
    
    def __getitem__(self, index):
        labels = torch.tensor(Norm(self.p[index], np.max(self.p), np.min(self.p))).float()
        angle = torch.tensor(Norm(self.angle[index], np.max(self.angle), np.min(self.angle))).float()
        x = torch.tensor(Norm(self.x[index], np.max(self.x), np.min(self.x))).float()
        y = torch.tensor(Norm(self.y[index], np.max(self.y), np.min(self.y))).float()
        z = torch.tensor(Norm(self.z[index], np.max(self.z), np.min(self.z))).float()
        v = torch.tensor(Norm(self.v[index], np.max(self.v), np.min(self.v))).float()
        Features = [angle, v, x, y, z]
        Features = torch.tensor(Features).float()
        return Features, labels

# gain_all_point_data_sphere_transiet()