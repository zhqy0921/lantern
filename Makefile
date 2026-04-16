CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -Wpedantic -std=c11

INCLUDES := -Isrc
LIB := liblantern.a
SRC := src/lantern_poly.c
OBJ := $(SRC:.c=.o)

TEST_BIN := tests/test_poly
TEST_OBJ := tests/test_poly.o

.PHONY: all clean test

all: $(LIB) $(TEST_BIN)

$(LIB): $(OBJ)
	ar rcs $@ $^

$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(TEST_OBJ) -L. -llantern

src/%.o: src/%.c src/lantern_poly.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

tests/%.o: tests/%.c src/lantern_poly.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(LIB) $(TEST_BIN)
