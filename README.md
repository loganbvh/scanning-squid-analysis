# scanning-squid-analysis

## Image analysis for scanning SQUID microscopy.

![scanning-squid-analysis GUI](screengrab.png)

### Requirements
- [Anaconda 3](https://www.anaconda.com/distribution/#download-section)

### Installation
- [Clone](https://help.github.com/en/articles/cloning-a-repository) or download this repository.

### Setup
#### Windows
- Edit the first line of `create_env.bat`, `remove_env.bat`, and `start_gui.bat` to read `call <path\to\your\Anaconda3>\Scripts\activate.bat`. For most users this is probably `C:\Users\<your-user>\Anaconda3\Scripts\activate.bat`. 
  - If you don't know where to find this, open the Anaconda Prompt and run `where python`. You should see something like `C:\Users\<your-user>\Anaconda3\python.exe` (see [here](https://docs.anaconda.com/anaconda/user-guide/tasks/integration/python-path/)). Then update the first line of the above .bat scripts with `call C:\Users\<your-user>\Anaconda3\Scripts\activate.bat`.
 - Double click `creat_env.bat` to create a `conda env` called `scanning-squid-analysis` with all of the required packages. You only have to do this once, when you first install the program. You can close this cmd windown once it reads `Press any key to continue...`.
 - You can now start the data analysis GUI by double clicking `start_gui.bat`.
 
#### Mac
 - Edit the second line of `create_env.command` and `remove_env.command` to read `source <path/to/your>/anaconda3/bin/activate`. For most users this is probably `/anaconda3/bin/activate` or `~/anaconda3/bin/activate`.
 - Edit the second line of `start_gui.command` to read `source <path/to/your>/anaconda3/bin/activate scanning-squid-analysis`.
 - Open Terminal and navigate to this repository (`cd <path/to/this/repo>` or `cd <space>` and drag/drop this directory into the Terminal window).
 - In Terminal, run `chmod u+x create_env.command remove_env.command start_gui.command` to give the terminal permission to run these files.
 - Double click `create_env.command` to create a `conda env` called `scanning-squid-analysis` with all of the required packages. You only have to do this once, when you first install the program. You can close this Terminal window once it reads `[Process completed]`.
 - You can now start the data analysis GUI by double clicking `start_gui.command`.
 
 ### Notes
 - If you wish to uninstall the program, double click `remove_env.bat` (Windows) or `remove_env.command` (Mac) to remove the `scanning-squid-analysis` `conda env`, then delete this repository.
 - The built-in IPython console has access to the following:
  - matplotlib.pyplot: `plt`
  - numpy: `np`
  - a dict of the current arrays in the form of [pint](https://pint.readthedocs.io/en/latest/) `Quantities`: `arrays`. For example, `arrays['MAG'].magnitude` will be the `MAG` array, with units `arrays['MAG'].units`
  - the current `qcodes.data.DataSet`: `dataset`