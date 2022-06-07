import pydot
(graph,) = pydot.graph_from_dot_file('draw_the_tree.dot')
graph.write_png('somefile.png')
