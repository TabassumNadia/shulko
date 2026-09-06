"""LangGraph wiring: nodes and edges. Built in the H+9 block."""

# TODO H+9
# from langgraph.graph import StateGraph, END
# from backend.graph.state import ShulkoState
#
# def build_graph():
#     g = StateGraph(ShulkoState)
#     g.add_node("extract", extract_node)
#     g.add_node("classify", classify_node)
#     g.add_node("duty", duty_node)
#     g.set_entry_point("extract")
#     g.add_edge("extract", "classify")
#     g.add_edge("classify", "duty")
#     g.add_edge("duty", END)
#     return g.compile()
