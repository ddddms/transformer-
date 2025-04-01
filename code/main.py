from train import *
from date_process import *
from module import *


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).to("cuda")
        data[:, 0] = 1
        src = data
        tgt = data
        yield Batch(src, tgt, 0)


def greedy_decoder(Model, Src, Src_mask, Max_len, start):
    encode = Model.encode(Src, Src_mask)
    ys = torch.ones(1, 1).fill_(start).type_as(Src.data)
    for i in range(Max_len - 1):
        out = Model.decode(encode, Src_mask, ys, subsequent_mask(ys.size(1)).type_as(Src.data))

        prob = Model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(Src.data).fill_(next_word)], dim=1)
    return ys


# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
#     for i in range(max_len - 1):
#         out = model.decode(memory, src_mask,
#                            ys,
#                            subsequent_mask(ys.size(1))
#                            .type_as(src.data))
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.data[0]
#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#     return ys


if __name__ == '__main__':
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = transformer(V, V, N=2)
    model.to("cuda")
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = torch.LongTensor([[1, 5, 3, 7, 2, 6, 4, 8, 9, 10]]).to("cuda")
    src_mask = torch.ones(1, 1, 10).to("cuda")
    print(greedy_decoder(model, src, src_mask, 10, 1))
