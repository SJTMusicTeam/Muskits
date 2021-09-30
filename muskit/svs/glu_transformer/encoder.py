class Encoder(nn.Module):
    """Encoder Network."""

    def __init__(
        self,
        phone_size,
        embed_size,
        hidden_size,
        dropout,
        GLU_num,
        num_layers=1,
        glu_kernel=3,
    ):
        """init."""
        # :param para: dictionary that contains all parameters
        super(Encoder, self).__init__()

        self.emb_phone = nn.Embedding(phone_size, embed_size)
        # full connected
        self.fc_1 = nn.Linear(embed_size, hidden_size)

        self.GLU_list = nn.ModuleList()
        for i in range(int(GLU_num)):
            self.GLU_list.append(
                module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)
            )
        # self.GLU =
        # module.GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)

        self.fc_2 = nn.Linear(hidden_size, embed_size)

    def forward(self, text_phone, pos=None):
        """forward."""
        # text_phone dim: [batch_size, text_phone_length]
        # output dim : [batch_size, text_phone_length, embedded_dim]

        # don't use pos in glu, but leave the field for uniform interface
        embedded_phone = self.emb_phone(text_phone)
        glu_in = self.fc_1(embedded_phone)

        batch_size = glu_in.shape[0]
        text_phone_length = glu_in.shape[1]
        embedded_dim = glu_in.shape[2]

        for glu in self.GLU_list:
            glu_out = glu(glu_in)
            glu_in = glu_out.reshape(batch_size, text_phone_length, embedded_dim)

        glu_out = self.fc_2(glu_in)

        out = embedded_phone + glu_out

        out = out * math.sqrt(0.5)
        return out, text_phone