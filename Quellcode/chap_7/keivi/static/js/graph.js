function computeCytoscapeGraph(kerasGraph, prefix) {
    return kerasGraph.class_name === 'Model'
        ? getModelCytoscapeGraph(kerasGraph, prefix)
        : getSequentialCytoscapeGraph(kerasGraph, prefix);
}

function getModelCytoscapeGraph(kerasGraph) {

    /**
     * Precompute this for performance
     */
    const subgraphMap = kerasGraph.config.layers.reduce(
        (currentMap, layer) => Object.assign(
            currentMap, {
                [layer.name]: isNodeSubgraph(layer) ? layer : false
            }
        ), {}
    );

    return [].concat(
        /**
         * Nodes
         */
        kerasGraph.config.layers.reduce(
            (nodes, layer) => (
                subgraphMap[layer.name]
                    ? nodes.concat(computeCytoscapeGraph(layer, layer.name))
                    : nodes.concat([
                        {
                            data: {
                                id: layer.name,
                                data: layer,
                                shape: getShapeForLayerClassName(layer.class_name),
                                color: layer.class_name === 'Conv2D' ? '#FFCC00' : '#FF0000'
                            }
                        }
                    ])
            ), []
        )
    ).concat(
        /**
         * Links
         */
        kerasGraph.config.layers.reduce((links, layer) => (
            links.concat(
                layer.inbound_nodes.length ? layer.inbound_nodes[0].map(
                    (inboundNode) => ({
                        data: {
                            id: layer.name + '-' + inboundNode[0],
                            source: subgraphMap[inboundNode[0]]
                                ? getSubgraphEnd(subgraphMap[inboundNode[0]]) : inboundNode[0],
                            target: subgraphMap[layer.name]
                                ? getSubgraphStart(subgraphMap[layer.name]) : layer.name
                        }
                    })
                ) : []
            )
        ), [])
    );
}

/* */
function getShapeForLayerClassName(class_name)
{
    if(class_name === 'Conv2D')
        return 'roundrectangle';
    if(class_name === 'MaxPooling2D')
        return "triangle";
    if(class_name === 'Flatten')
        return "square";
    if(class_name === 'BatchNormalization')
        return "star";

    return 'ellipse'
} 

/* */
function getColorForLayerClassName(class_name)
{
    if(class_name === 'Conv2D')
        return '#FFCC00';
    if(class_name === 'MaxPooling2D')
        return "lightgrey";
    if(class_name === 'Flatten')
        return "darkgrey";
    if(class_name === 'BatchNormalization')
        return "aqua";
    return 'orange'
}


function getSequentialCytoscapeGraph(kerasGraph, prefix = '') {
    return [].concat(
        /**
         * Nodes
         */
        kerasGraph.config.layers.map( // Config.layers instead of config.map
            (layer) => ({
                data: {
                    id: layer.config.name,
                    data: layer.config,
                    shape: getShapeForLayerClassName(layer.class_name),
                    // === 'Conv2D' ? 'roundrectangle' : 'ellipse',
                    color: getColorForLayerClassName(layer.class_name)
                }
                
            })
        )
    ).concat(
        /**
         * Links
         */
        kerasGraph.config.layers.reduce((linksData, layer) => (
            {
                links: linksData.links.concat(
                    linksData.lastNodeId
                    ? [
                        {
                            data: {
                                id: layer.config.name + '-' + linksData.lastNodeId,
                                source: linksData.lastNodeId,
                                target: layer.config.name
                            }
                        }
                    ] : []
                ),
                lastNodeId: layer.config.name
            }
        ), {
            links: [],
            lastNodeId: ''
        }).links
    );

}

function getSubgraphStart(subgraph) {
    return subgraph.class_name === 'Sequential'
        ? subgraph.name + '.' + subgraph.config[0].config.name : null;
}

function getSubgraphEnd(subgraph) {
    return subgraph.class_name === 'Sequential'
        ? subgraph.name + '.' + subgraph.config[subgraph.config.length - 1].config.name : null;
}

function isNodeSubgraph(layer) {
    return ['Sequential', 'Model'].indexOf(layer.class_name) !== -1;
}

function buildGraph(kerasModel) {
    const cyGraph = cytoscape({
      container: document.getElementById("graphvisContainer"),
      elements: computeCytoscapeGraph(kerasModel),
      layout: {
        name: 'dagre',
        rankDir: 'TD'
      },
      fit: true,
      userZoomingEnabled: false,
      userPanningEnabled: false,
      style: [{
          selector: 'node',
          style: {
            'content': 'data(id)',
            'text-opacity': 1,
            'font-family': "Verdana",
            'font-size': "16px",
            'background-color': 'data(color)',
            //'background-color': "#FFCC00",
            'label': 'data(id)',
            'border-color': "#666",
            "border-width": 5,
            'text-valign': 'center',
            'text-halign': 'right',
            'text-margin-x': 25,

            shape: 'data(shape)',
            width: 'mapData(data.config.nb_col, 1, 20, 100, 250)',
            height: 'mapData(data.config.nb_row, 1, 20, 100, 250)',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 5,
            'target-arrow-shape': 'triangle',
            'target-arrow-color': "#CCC",
            'curve-style': 'bezier',
            'line-color': '#666'
          }
        }
      ]
    });


    cyGraph.nodes().on("click", (clickEvent) => {
      
        nodeData = clickEvent.target._private.data.data;
        $("#node_info").text(JSON.stringify(nodeData, null, 2))
        // console.log(clickEvent.target._private.data.data);
    })
  }