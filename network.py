import streamlit.components.v1 as components
import pandas as pd 
import networkx as nx
from pyvis import network as net
from pyvis.network import Network

rel = pd.read_csv("data/champion_releated.csv")

rel_melt = rel.melt(id_vars="이름")
i= 1
for i in range(9) :
    rel_melt = rel_melt.replace(f'관련 챔피언 {i}', 9-i)
relrel = rel_melt.drop(rel_melt.loc[rel_melt['value']=='-'].index)

got_net = net.Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True)

# set the physics layout of the network
got_net.barnes_hut()
# got_data = pd.read_csv('data/champion_releated.csv')

sources = relrel['이름']
targets = relrel['value']
weights = relrel['variable']


for src, dst, w in zip(sources, targets, weights):
    got_net.add_node(src, src, title=src, color="white")
    got_net.add_node(dst, dst, title=dst, color="white")
    
#     if w == 8:
#         color = "red"
#     elif w == 7:
#         color = "orange"
#     elif w == 6:
#         color = "yellow"
#     elif w == 5:
#         color = "green"
#     elif w == 4:
#         color = "blue"
#     elif w <= 5:
#         color = "purple"    

    if w == 8:
        color = "#D4F37B"
    else: color = "#5CC1B1"   
        
    got_net.add_edge(src, dst, value=w, color=color)

neighbor_map = got_net.get_adj_list()

# add neighbor data to node hover data
for node in got_net.nodes:
    node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
    node['value'] = len(neighbor_map[node['id']])

    
got_net.set_options("""
const options = {
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -80000,
      "springLength": 570,
      "springConstant": 0.001
    },
    "minVelocity": 0.75
  },

  "nodes": {
    "borderWidth": null,
    "borderWidthSelected": null,
    "opacity": null,
    "font": {
      "size": 150
    },
    "size": null
  },
  
  "edges": {
    "arrows": {
      "to": {
        "enabled": true
      },
      "middle": {
        "enabled": true
      },
      "from": {
        "enabled": true
      }
    },
    "color": {
      "inherit": true
    },
    "selfReferenceSize": null,
    "selfReference": {
      "angle": 0.7853981633974483
    },
    "smooth": {
      "forceDirection": "none"
    }
  }

}
""")

for name in rel_melt[(rel_melt['variable']==8) & (rel_melt['value']=="-")]["이름"]:
    got_net.add_node(name, size=25, label=name, title=name)
#     got_net.add_node(name, name, title=name, color="white")
# got_net.show_buttons(filter_=['physics'])
# got_net.show_buttons(filter_=['nodes'])
# got_net.show_buttons(filter_=['edges'])
got_net.save_graph("example.html")
file = open("example.html","r",encoding="UTF-8")
components.html(file.read(),height=800)
