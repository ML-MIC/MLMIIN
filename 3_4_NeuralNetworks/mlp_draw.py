def mlp_draw(model, orientation = "v", figsize = (5, 12), add_labels = True, label_pos = 5/7, label_size = 7, alpha=1):
    
    # Create a directed graph
    G = nx.DiGraph()
    
     
    options = {
    'node_color':'black',
    'node_size':1,
    'alpha':0.2}
    
    # Add nodes
    for i in range(model.n_layers_ - 1):
        # print("topj =" , model.coefs_[i].shape[0])
        for j in range(model.coefs_[i].shape[0]):
            # print("Node ", (i, j), " added")
            G.add_node((i, j), layer=i)
    # For the output layer neurons
    i = model.n_layers_ - 1
    out_n = model.coefs_[model.n_layers_ - 2].shape[1]
    for j in range(out_n):
            # print("Node ", (i, j), " added")
            G.add_node((i, j), layer=i)
    # G.add_node((model.n_layers_ - 1, 0), layer=model.n_layers_ - 1)
    # print("output")
    # print("Node ", (model.n_layers_ - 1, 0), " added")
    # print("----------")


    # Add edges
    for i in range(model.n_layers_ - 1):
        for j in range(model.coefs_[i].shape[0]):
            for k in range(model.coefs_[i].shape[1]):
                # print(f"(i, j, k) = {(i, j, i+1, k)}")
                wgt = np.round(model.coefs_[i][j, k], 3)
                G.add_edge((i, j), (i + 1, k), weight=wgt)
                # print(f"Added edge from {(i, j)} to {(i + 1, k)}")
                # G.add_edge((i, j), (i + 1, k), weight=1)
                
    # Define the number of layers
    num_layers = model.n_layers_
    max_layer_size = max([s.shape[1] for s in model.coefs_])
    # print("num_layers =", num_layers)
    # print("max_layer_size =", max_layer_size, "\n-------\n")


    # Define the horizontal space between layers
    horizontal_space = 2

    # Define the y-coordinate for each layer
    layer_y = np.linspace(0.1, 1.9, num_layers)

    # Create a dictionary to store the positions of nodes
    pos = {}


    # Add nodes
    for i in range(num_layers - 1):
        # print(i, "\n", model.coefs_[i])
        layer_size = model.coefs_[i].shape[0]
        offset = max_layer_size - layer_size
        for j in range(layer_size):
            pos[(i, j)] = (offset + j * horizontal_space, layer_y[i])
            # print(f"pos[{(i, j)}] =", pos[(i, j)])
        # print("\n------------------\n")


    # For the output layer neurons
    layer_size = model.coefs_[model.n_layers_ - 2].shape[1]
    offset = max_layer_size - layer_size
    for j in range(layer_size):
        pos[(num_layers - 1, j)] = (offset + j * horizontal_space, layer_y[-1])
        
    # print(f"pos[({num_layers - 1}, 0)] =", pos[(num_layers - 1, 0)])            

    if orientation == "h":
        pos = {k:(v[1], v[0]) for k, v in pos.items()}
        figsize = (figsize[1], figsize[0])

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=[n[0] for n in G.nodes()], alpha = alpha, cmap='Pastel1')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if add_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=label_pos, font_size=label_size)
    plt.title('Neural Network Architecture')
    plt.show()




