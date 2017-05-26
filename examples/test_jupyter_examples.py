r"""
Automatically run Jupyter notebooks cell-by-cell.


Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

from __future__ import print_function
import os, sys
import jupyter_client
import nbformat as ipy
import pytest


def run_notebook(nb, name):
    print(name, end=" ")
    k = jupyter_client.KernelManager()
    k.start_kernel(stderr=open(os.devnull, 'w'))
    assert k.is_alive()
    c = k.blocking_client()
    c.execute("import sys; sys.path.append('..')")
    for i,cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            print(i, end=" ")
            c.execute(cell.source, stop_on_error=True, silent=False)
            reply = c.get_shell_msg()
            content = reply['content']
            if reply['msg_type'] == "execute_reply" and content['status'] != 'ok':
                try:
                    ename = content['ename']
                    evalue = content['evalue']
                except:
                    raise RuntimeError(
                        "Unknown error in cell {} of {}".format(i, name))
                    
                raise RuntimeError(
                    "Notebook error {} {} at cell {} of {}".format(
                        ename, evalue, i, name))
            
            if not k.is_alive():
                raise RuntimeError(
                    "Unknown error in cell {} of {}".format(i, name))


    print()    

    assert k.is_alive()
    k.shutdown_kernel()
    del k
    pass


class TestJupyterNotebooks:
    def setup(self):
        os.chdir("examples")
        self.filenames = filter(lambda x:x.endswith("ipynb"), os.listdir("."))

    def teardown(self):
        del self.filenames
        os.chdir("..")

    def setup_method(self, function):
        pass
    
    def teardown_method(self, function):
        pass
    
    def test_all_notebooks(self):
        for fn in self.filenames:
            with open(fn) as f:
                nb = ipy.read(f, as_version=4) 
            run_notebook(nb, fn) 




if __name__ == '__main__':
    if len(sys.argv) <= 1:
        pass
    else:
        for ipynb in sys.argv[1:]:
            with open(ipynb, encoding='utf8') as f:
                nb = ipy.read(f, as_version=4) 
            run_notebook(nb, ipynb)
