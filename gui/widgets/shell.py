from IPython.lib import guisupport

try:
    from qtconsole.inprocess import QtInProcessKernelManager
    from qtconsole.rich_jupyter_widget import RichJupyterWidget as RichIPythonWidget
except ImportError:
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.qt.inprocess import QtInProcessKernelManager


class QJupyterWidget(RichIPythonWidget):
    """Convenience class for a live Jupyter console widget."""

    def __init__(self, custom_banner=None, *args, **kwargs):
        if custom_banner is not None:
            self.banner = custom_banner
        super().__init__(*args, **kwargs)
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = "qt"
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()

        self.exit_requested.connect(stop)

    def push_variables(self, variable_dict):
        """Given a dictionary containing name / value pairs,
        push those variables to the Jupyter console widget.
        """
        self.kernel_manager.kernel.shell.push(variable_dict)

    def clear(self):
        """Clears the terminal."""
        self._control.clear()

    def print_text(self, text):
        """Prints some plain text to the console."""
        self._append_plain_text(text)

    def execute_command(self, command):
        """Execute a command in the console widget."""
        self._execute(command, False)
