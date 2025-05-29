class Node:
    def __init__(self, is_final=False):
        self.edges = dict()
        self.is_final = is_final
    def add_child(self, letter):
        self.edges[letter] = Node()
        return self.edges[letter]
    def has_child(self, letter):
        return letter in self.edges
    def get_child(self, letter):
        return self.edges.get(letter)
    def to_str(self, letter="", offset=0, to_make_offset=True):
        s = "_" * offset if to_make_offset else ""
        s += letter
        offset += len(letter)
        # s = " "*offset+letter
        if self.is_final:
            s += "F"
            offset += 1
        if len(self.edges) > 0:
            for i, (letter, child) in enumerate(self.edges.items()):
                if i == 0:
                    s += "_"
                s += child.to_str(letter=letter, offset=offset+1,
                                  to_make_offset=(i>0))
        else:
            s += "\n"
        return s

class Trie:
    def __init__(self, words=None):
        self.root = Node()
        # self.nodes = [self.root]
        if words is not None:
            for word in words:
                self.add(word)
    def _add_path(self, curr, word):
        for a in word:
            curr = curr.add_child(a)
        return curr
    def add(self, word):
        curr = self.root
        for i, a in enumerate(word):
            child = curr.get_child(a)
            if child is None:
                curr = self._add_path(curr, word[i:])
                break
            curr = child
        curr.is_final = True
    def __contains__(self, word):
        curr = self.root
        for i, a in enumerate(word):
            child = curr.get_child(a)
            if child is None:
                return False
            curr = child
        return curr.is_final
    def __str__(self):
        return self.root.to_str()