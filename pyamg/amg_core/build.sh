clang \
    -Wno-unused-result \
    -Wsign-compare \
    -Wunreachable-code \
    -fno-common \
    -dynamic \
    -DNDEBUG \
    -g \
    -fwrapv \
    -O3 \
    -Wall \
    -I/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include \
    -I/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/System/Library/Frameworks/Tk.framework/Versions/8.5/Headers \
    -DVERSION_INFO="4.0.0.dev0+a35015b" \
    -UNDEBUG \
    -I/Users/lukeo/repos/pyamg/.eggs/pybind11-2.6.0-py3.8.egg/pybind11/include \
    -I/Users/lukeo/repos/pyamg/.eggs/pybind11-2.6.0-py3.8.egg/pybind11/include \
    -I/Users/lukeo/.virtualenvs/pyamg1/include \
    -I/usr/local/opt/python@3.8/Frameworks/Python.framework/Versions/3.8/include/python3.8 \
    -I/Users/lukeo/.virtualenvs/pyamg1/lib/python3.8/site-packages/numpy/core/include \
    -c graph_bind.cpp \
    -o graph_bind.o \
    -stdlib=libc++ \
    -mmacosx-version-min=10.7 \
    -std=c++14 \
    -fvisibility=hidden

clang++ -bundle -undefined dynamic_lookup \
        -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk \
        graph_bind.o \
        -o graph.cpython-38-darwin.so \
        -stdlib=libc++ -mmacosx-version-min=10.7
