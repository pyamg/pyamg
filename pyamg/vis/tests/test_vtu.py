""" Test 1, 2, 8 element meshes writing the vtu file.  Validate as well-formed
xml."""

import tempfile

import xml.parsers.expat
from numpy import array, uint32

from pyamg.vis import write_vtu

from numpy.testing import TestCase


class TestWriteVtu(TestCase):
    def setUp(self):
        cases = []

        class mesh:
            file_name = tempfile.mktemp()
            Verts = None
            E2V = None
            pdata = None
            cdata = None
        mesh = mesh()

        # 1 triangle
        mesh.Verts = array([[0.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 1.0]])
        E2V = array([[0, 2, 1]], uint32)
        mesh.Cells = {5: E2V}
        mesh.pdata = None
        mesh.cdata = None
        cases.append(mesh)

        # 2 triangles
        mesh.Verts = array([[0.0, 0.0],
                           [1.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 1.0]])
        E2V = array([[0, 3, 2],
                     [0, 1, 3]], uint32)
        mesh.Cells = {5: E2V}
        mesh.pdata = None
        mesh.cdata = None
        cases.append(mesh)

        # 8 triangles
        mesh.Verts = array([[0.0, 0.0],
                           [1.0, 0.0],
                           [2.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 1.0],
                           [2.0, 1.0],
                           [0.0, 2.0],
                           [1.0, 2.0],
                           [2.0, 2.0]])
        E2V = array([[0, 4, 3],
                     [0, 1, 4],
                     [1, 5, 4],
                     [1, 2, 5],
                     [3, 7, 6],
                     [3, 4, 7],
                     [4, 8, 7],
                     [4, 5, 8]], uint32)
        mesh.Cells = {5: E2V}
        mesh.pdata = None
        mesh.cdata = None
        cases.append(mesh)

        self.cases = cases

    def test_xml(self):
        for mesh in self.cases:
            write_vtu(Verts=mesh.Verts, Cells=mesh.Cells,
                      pdata=mesh.pdata, cdata=mesh.cdata,
                      fname=mesh.file_name)

            try:
                parser = xml.parsers.expat.ParserCreate()
                with open(mesh.file_name, 'rb') as f:
                    parser.ParseFile(f)
            except Exception as ex:
                assert False, 'problem: %s' % (ex)
