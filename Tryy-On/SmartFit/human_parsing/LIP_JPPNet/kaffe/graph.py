from google.protobuf import text_format

from .caffe import get_caffe_resolver
from .errors import KaffeError, print_stderr
from .layers import LayerAdapter, LayerType, NodeKind, NodeDispatch
from .shapes import TensorShape

class Node(object):

    def __init__(self, name, kind, layer=None):
        self.name = name
        self.kind = kind
        self.layer = LayerAdapter(layer, kind) if layer else None
        self.parents = []
        self.children = []
        self.data = None
        self.output_shape = None
        self.metadata = {}

    def add_parent(self, parent_node):
        assert parent_node not in self.parents
        self.parents.append(parent_node)
        if self not in parent_node.children:
            parent_node.children.append(self)

    def add_child(self, child_node):
        assert child_node not in self.children
        self.children.append(child_node)
        if self not in child_node.parents:
            child_node.parents.append(self)

    def get_only_parent(self):
        if len(self.parents) != 1:
            raise KaffeError('Node (%s) expected to have 1 parent. Found %s.' %
                             (self, len(self.parents)))
        return self.parents[0]

    @property
    def parameters(self):
        if self.layer is not None:
            return self.layer.parameters
        return None

    def __str__(self):
        return '[%s] %s' % (self.kind, self.name)

    def __repr__(self):
        return '%s (0x%x)' % (self.name, id(self))


class Graph(object):

    def __init__(self, nodes=None, name=None):
        self.nodes = nodes or []
        self.node_lut = {node.name: node for node in self.nodes}
        self.name = name

    def add_node(self, node):
        self.nodes.append(node)
        self.node_lut[node.name] = node

    def get_node(self, name):
        try:
            return self.node_lut[name]
        except KeyError:
            raise KaffeError('Layer not found: %s' % name)

    def get_input_nodes(self):
        return [node for node in self.nodes if len(node.parents) == 0]

    def get_output_nodes(self):
        return [node for node in self.nodes if len(node.children) == 0]

    def topologically_sorted(self):
        sorted_nodes = []
        unsorted_nodes = list(self.nodes)
        temp_marked = set()
        perm_marked = set()

        def visit(node):
            if node in temp_marked:
                raise KaffeError('Graph is not a DAG.')
            if node in perm_marked:
                return
            temp_marked.add(node)
            for child in node.children:
                visit(child)
            perm_marked.add(node)
            temp_marked.remove(node)
            sorted_nodes.insert(0, node)

        while len(unsorted_nodes):
            visit(unsorted_nodes.pop())
        return sorted_nodes

    def compute_output_shapes(self):
        sorted_nodes = self.topologically_sorted()
        for node in sorted_nodes:
            node.output_shape = TensorShape(*NodeKind.compute_output_shape(node))

    def replaced(self, new_nodes):
        return Graph(nodes=new_nodes, name=self.name)

    def transformed(self, transformers):
        graph = self
        for transformer in transformers:
            graph = transformer(graph)
            if graph is None:
                raise KaffeError('Transformer failed: {}'.format(transformer))
            assert isinstance(graph, Graph)
        return graph

    def __contains__(self, key):
        return key in self.node_lut

    def __str__(self):
        hdr = '{:<20} {:<30} {:>20} {:>20}'.format('Type', 'Name', 'Param', 'Output')
        s = [hdr, '-' * 94]
        for node in self.topologically_sorted():
            # If the node has learned parameters, display the first one's shape.
            # In case of convolutions, this corresponds to the weights.
            data_shape = node.data[0].shape if node.data else '--'
            out_shape = node.output_shape or '--'
            s.append('{:<20} {:<30} {:>20} {:>20}'.format(node.kind, node.name, data_shape,
                                                          tuple(out_shape)))
        return '\n'.join(s)


class GraphBuilder(object):
    '''Constructs a model graph from a Caffe protocol buffer definition.'''

    def __init__(self, def_path, phase='test'):
        '''
        def_path: Path to the model definition (.prototxt)
        data_path: Path to the model data (.caffemodel)
        phase: Either 'test' or 'train'. Used for filtering phase-specific nodes.
        '''
        self.def_path = def_path
        self.phase = phase
        self.load()

    def load(self):
        '''Load the layer definitions from the prototxt.'''
        self.params = get_caffe_resolver().NetParameter()
        with open(self.def_path, 'rb') as def_file:
            text_format.Merge(def_file.read(), self.params)

    def filter_layers(self, layers):
        '''Filter out layers based on the current phase.'''
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer in layers:
            phase = self.phase
            if len(layer.include):
                phase = phase_map[layer.include[0].phase]
            if len(layer.exclude):
                phase = phase_map[1 - layer.include[0].phase]
            exclude = (phase != self.phase)
            if (not exclude) and (phase == 'test'):
                exclude = (layer.type == LayerType.Dropout)
            if not exclude:
                filtered_layers.append(layer)
                # Guard against dupes.
                assert layer.name not in filtered_layer_names
                filtered_layer_names.add(layer.name)
        return filtered_layers

    def make_node(self, layer):
        '''Create a graph node for the given layer.'''
        kind = NodeKind.map_raw_kind(layer.type)
        if kind is None:
            raise KaffeError('Unknown layer type encountered: %s' % layer.type)
        return Node(layer.name, kind, layer=layer)

    def make_input_nodes(self):
        '''
        Create data input nodes.

        This method is for old-style inputs, where the input specification
        was not treated as a first-class layer in the prototext.
        Newer models use the "Input layer" type.
        '''
        nodes = [Node(name, NodeKind.Data) for name in self.params.input]
        if len(nodes):
            input_dim = map(int, self.params.input_dim)
            if not input_dim:
                if len(self.params.input_shape) > 0:
                    input_dim = map(int, self.params.input_shape[0].dim)
                else:
                    raise KaffeError('Dimensions for input not specified.')
            for node in nodes:
                node.output_shape = tuple(input_dim)
        return nodes

    def build(self):
        '''
        Builds the graph from the Caffe layer definitions.
        '''
        layers = self.params.layers or self.params.layer
        layers = self.filter_layers(layers)
        nodes = self.make_input_nodes()
        nodes += [self.make_node(layer) for layer in layers]
        graph = Graph(nodes=nodes, name=self.params.name)
        
        node_outputs = {}
        for layer in layers:
            node = graph.get_node(layer.name)
            for input_name in layer.bottom:
                assert input_name != layer.name
                parent_node = node_outputs.get(input_name)
                if (parent_node is None) or (parent_node == node):
                    parent_node = graph.get_node(input_name)
                node.add_parent(parent_node)
            if len(layer.top)>1:
                raise KaffeError('Multiple top nodes are not supported.')
            for output_name in layer.top:
                if output_name == layer.name:
                    continue
                node_outputs[output_name] = node

        graph.compute_output_shapes()
        return graph


class NodeMapper(NodeDispatch):

    def __init__(self, graph):
        self.graph = graph

    def map(self):
        nodes = self.graph.topologically_sorted()
        input_nodes = self.graph.get_input_nodes()
        nodes = [t for t in nodes if t not in input_nodes]
        chains = []
        for node in nodes:
            attach_to_chain = None
            if len(node.parents) == 1:
                parent = node.get_only_parent()
                for chain in chains:
                    if chain[-1] == parent:
                        attach_to_chain = chain
                        break
            if attach_to_chain is None:
                attach_to_chain = []
                chains.append(attach_to_chain)
            attach_to_chain.append(node)
        mapped_chains = []
        for chain in chains:
            mapped_chains.append(self.map_chain(chain))
        return self.commit(mapped_chains)

    def map_chain(self, chain):
        return [self.map_node(node) for node in chain]

    def map_node(self, node):
        map_func = self.get_handler(node.kind, 'map')
        mapped_node = map_func(node)
        assert mapped_node is not None
        mapped_node.node = node
        return mapped_node

    def commit(self, mapped_chains):
        raise NotImplementedError('Must be implemented by subclass.')
