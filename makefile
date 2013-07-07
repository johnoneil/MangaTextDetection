all:
	g++ -o opencvtest opencvtest.cpp `pkg-config opencv --cflags --libs`
	g++ -o text_link_text text_link_test.cpp `pkg-config opencv --cflags --libs`
