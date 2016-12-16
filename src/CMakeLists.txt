FILE(GLOB src_list *.cpp *.h)

add_executable(TD_BEM_Stability ${src_list})


target_link_libraries(TD_BEM_Stability libpetsc.dylib libmpi.dylib libslepc.3.7.3.dylib)




