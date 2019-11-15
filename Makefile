sudCXX=g++

LIBDIR = -L~/usr/lib  -L/usr/local/lib 

DEBUG=
OPTIM=-O2
# BLASOARMA_MACOS= -lblas -llapack -DARMA_DONT_USE_WRAPPER -framework Accelerate  
BLASOARMA= -lblas -llapack
INCLUDE= -I/mnt/d/WorkspaceC++/include/
CFLAGS= $(DEBUG) $(OPTIM) $(INCLUDE) -std=c++11 -fpermissive -Wall -w $(BLASOARMA)

LIBS= $(BLASOARMA) -lm -larmadillo -w 
SRCDIR=./src
OBJDIR=./obj

SRCS := $(shell find . -name "*.cpp")
OBJS := $(patsubst %.cpp, %.o, $(SRCS))
OBJS := $(patsubst $(SRCDIR)%, $(OBJDIR)%, $(OBJS))
DEPS := $(shell find . -name "*.hpp")

$(info $$SRCS is [${SRCS}])

NAME=Solver

main: $(OBJS) $(OBJDIR)/main.o
	$(CXX) -o $(NAME) $(OBJ) $(OBJDIR)/main.o $(CFLAGS) $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

run: main
	./Solver

.PHONY: clean

clean:
	rm -f src/*.o *~ ./lib/lib$(NAME).a ./NMFSolver






