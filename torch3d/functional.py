import torch


def meshgrid2d(
    start,
    end,
    steps,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
):
    xx = torch.linspace(
        start,
        end,
        steps,
        out=None,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=False,
    )
    yy = torch.linspace(
        start,
        end,
        steps,
        out=None,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=False,
    )
    xx, yy = torch.meshgrid(xx, yy)
    g = torch.zeros(
        2,
        steps ** 2,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
    g[0] = xx.reshape(-1)
    g[1] = yy.reshape(-1)
    return g
