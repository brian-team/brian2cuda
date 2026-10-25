OBJECTS := customUpdate.o neuronUpdate.o synapseUpdate.o init.o runner.o 
CXXFLAGS := -c -fPIC -std=c++11 -MMD -MP 
LINKFLAGS := -shared 


DEPS := $(OBJECTS:.o=.d)

.PHONY: all clean

all: librunner.so

librunner.so: $(OBJECTS)
	@$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)

-include $(DEPS)

%.o: %.cc %.d
	@$(CXX) $(CXXFLAGS) -o $@ $<

%.d: ;

clean:
	@rm -f $(OBJECTS) $(DEPS) librunner.so
