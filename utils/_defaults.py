__all__ = ["default_to"]


def default_to(*vals, typecast: bool = False):
    """Get the first non-None value, right to left order.

    Most "default" value goes first.
    """
    if len(vals) == 0:
        return None
    typecls = type(vals[0])
    if len(vals) == 1:
        return vals[0]
    for val in reversed(vals):
        if val is not None:
            if typecast:
                return typecls(val)
            return val
