#!/usr/bin/env python3
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')

print(roberta.fill_mask('The first Star wars movie came out in <mask>', topk=10))
# [('The first Star wars movie came out in 1977', 0.9504708051681519, ' 1977'), ('The first Star wars movie came out in 1978', 0.009986862540245056, ' 1978'), ('The first Star wars movie came out in 1979', 0.009574787691235542, ' 1979')]

# roberta.fill_mask('Vikram samvat calender is official in <mask>', topk=3)
# # [('Vikram samvat calender is official in India', 0.21878819167613983, ' India'), ('Vikram samvat calender is official in Delhi', 0.08547237515449524, ' Delhi'), ('Vikram samvat calender is official in Gujarat', 0.07556215673685074, ' Gujarat')]

# roberta.fill_mask('<mask> is the common currency of the European Union', topk=3)
# # [('Euro is the common currency of the European Union', 0.9456493854522705, 'Euro'), ('euro is the common currency of the European Union', 0.025748178362846375, 'euro'), ('€ is the common currency of the European Union', 0.011183084920048714, '€')]
