unset path
unset KERAS_HOME
set KERAS_HOME=%userprofile%\.keras
rem Если может работать с компилятором указать его папку в path
set path=B:\msys64\mingw64\bin\A_win_python\Python373;
start cmd /K python app.py
