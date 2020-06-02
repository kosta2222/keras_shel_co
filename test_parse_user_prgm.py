from unittest import TestCase
from app import parse_user_prgm


class TestParse_user_prgm(TestCase):
    def test_parse_user_prgm(self):
        list_:list=None
        list_targ=['cr_nn','fit','sav_model_wei']
        progr=\
        """cr_nn;fit;sav_model_wei"""

        # self.fail()
        list_=parse_user_prgm(progr)
        for i in range(len(list_targ)):
            el=list_[i]
            el_t=list_targ[i]
            self.assertEqual(el_t, el)

