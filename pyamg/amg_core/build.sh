clang -Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -g -fwrapv -O3 -Wall \
      -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk \
      -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk/usr/include \
      -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk/System/Library/Frameworks/Tk.framework/Versions/8.5/Headers -DVERSION_INFO="4.0.0" \
      -I/Users/lukeo/repos/pyamg/.eggs \
      -I/Users/lukeo/repos/pyamg/.eggs \
      -I/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/include/python3.7m \
      -I/Users/lukeo/.virtualenvs/lloyd1/lib/python3.7/site-packages/numpy/core/include \
      -c graph_bind.cpp \
      -o graph_bind.o \
      -stdlib=libc++ \
      -mmacosx-version-min=10.7 \
      -std=c++14 \
      -fvisibility=hidden

clang++ -bundle -undefined dynamic_lookup \
        -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk \
        graph_bind.o \
        -o graph.cpython-37m-darwin.so \
        -stdlib=libc++ -mmacosx-version-min=10.7
