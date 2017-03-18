objects = example.o layer_function.o activation_function.o 
CC = gcc

wave_cnn : $(objects)
	$(CC) -o wave_cnn $(objects) -lm
	
example.o : global.h layer_function.h
layer_function.o : global.h layer.h activation_function.h
activation_function.o : global.h activation_function.h

	
.PHONY : clean
clean :
	-rm wave_cnn $(objects)
