call path\to\anaconda3\Scripts\activate.bat
call activate scanning-squid-analysis
cd ..
python -m scanning-squid-analysis.gui.window
call conda deactivate
pause