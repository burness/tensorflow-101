#-*-coding:utf-8-*-
import jieba
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
import re
import traceback
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# from ..config import *
WORD_DICT = "/Users/burness/git_repository/dl_opensource/nlp/oxford-cs-deepnlp-2017/practical-2/data/origin_data/t_tag_infos.txt"


class cutDoc:
    """ cut_doc: cut the document
    """

    def __init__(self, mode="default"):
        self.mode = mode
        self.stop_words = [
            u"一",u"一下",u"一些",u"一切",u"一则",u"一天",u"一定",u"一方面",u"一旦",u"一时",u"一来",
            u"一样",u"一次",u"一片",u"一直",u"一致",u"一般",u"一起",u"一边",u"一面",u"万一",u"上下",
            u"上升",u"上去",u"上来",u"上述",u"上面",u"下列",u"下去",u"下来",u"下面",u"不一",u"不久",
            u"不仅",u"不会",u"不但",u"不光",u"不单",u"不变",u"不只",u"不可",u"不同",u"不够",u"不如",
            u"不得",u"不怕",u"不惟",u"不成",u"不拘",u"不敢",u"不断",u"不是",u"不比",u"不然",u"不特",
            u"不独",u"不管",u"不能",u"不要",u"不论",u"不足",u"不过",u"不问",u"与",u"与其",u"与否",
            u"与此同时",u"专门",u"且",u"两者",u"严格",u"严重",u"个",u"个人",u"个别",u"中小",u"中间",
            u"丰富",u"临",u"为",u"为主",u"为了",u"为什么",u"为什麽",u"为何",u"为着",u"主张",u"主要",
            u"举行",u"乃",u"乃至",u"么",u"之",u"之一",u"之前",u"之后",u"之後",u"之所以",u"之类",
            u"乌乎",u"乎",u"乘",u"也",u"也好",u"也是",u"也罢",u"了",u"了解",u"争取",u"于",u"于是",
            u"于是乎",u"云云",u"互相",u"产生",u"人们",u"人家",u"什么",u"什么样",u"什麽",u"今后",u"今天",
            u"今年",u"今後",u"仍然",u"从",u"从事",u"从而",u"他",u"他人",u"他们",u"他的",u"代替",
            u"以",u"以上",u"以下",u"以为",u"以便",u"以免",u"以前",u"以及",u"以后",u"以外",u"以後",
            u"以来",u"以至",u"以至于",u"以致",u"们",u"任",u"任何",u"任凭",u"任务",u"企图",u"伟大",
            u"似乎",u"似的",u"但",u"但是",u"何",u"何况",u"何处",u"何时",u"作为",u"你",u"你们",
            u"你的",u"使得",u"使用",u"例如",u"依",u"依照",u"依靠",u"促进",u"保持",u"俺",u"俺们",
            u"倘",u"倘使",u"倘或",u"倘然",u"倘若",u"假使",u"假如",u"假若",u"做到",u"像",u"允许",
            u"充分",u"先后",u"先後",u"先生",u"全部",u"全面",u"兮",u"共同",u"关于",u"其",u"其一",
            u"其中",u"其二",u"其他",u"其余",u"其它",u"其实",u"其次",u"具体",u"具体地说",u"具体说来",
            u"具有",u"再者",u"再说",u"冒",u"冲",u"决定",u"况且",u"准备",u"几",u"几乎",u"几时",u"凭",
            u"凭借",u"出去",u"出来",u"出现",u"分别",u"则",u"别",u"别的",u"别说",u"到",u"前后",
            u"前者",u"前进",u"前面",u"加之",u"加以",u"加入",u"加强",u"十分",u"即",u"即令",u"即使",
            u"即便",u"即或",u"即若",u"却不",u"原来",u"又",u"及",u"及其",u"及时",u"及至",u"双方",
            u"反之",u"反应",u"反映",u"反过来",u"反过来说",u"取得",u"受到",u"变成",u"另",u"另一方面",
            u"另外",u"只是",u"只有",u"只要",u"只限",u"叫",u"叫做",u"召开",u"叮咚",u"可",u"可以",
            u"可是",u"可能",u"可见",u"各",u"各个",u"各人",u"各位",u"各地",u"各种",u"各级",u"各自",
            u"合理",u"同",u"同一",u"同时",u"同样",u"后来",u"后面",u"向",u"向着",u"吓",u"吗",u"否则",
            u"吧",u"吧哒",u"吱",u"呀",u"呃",u"呕",u"呗",u"呜",u"呜呼",u"呢",u"周围",u"呵",u"呸",
            u"呼哧",u"咋",u"和",u"咚",u"咦",u"咱",u"咱们",u"咳",u"哇",u"哈",u"哈哈",u"哉",u"哎",
            u"哎呀",u"哎哟",u"哗",u"哟",u"哦",u"哩",u"哪",u"哪个",u"哪些",u"哪儿",u"哪天",u"哪年",
            u"哪怕",u"哪样",u"哪边",u"哪里",u"哼",u"哼唷",u"唉",u"啊",u"啐",u"啥",u"啦",u"啪达",
            u"喂",u"喏",u"喔唷",u"嗡嗡",u"嗬",u"嗯",u"嗳",u"嘎",u"嘎登",u"嘘",u"嘛",u"嘻",u"嘿",
            u"因",u"因为",u"因此",u"因而",u"固然",u"在",u"在下",u"地",u"坚决",u"坚持",u"基本",
            u"处理",u"复杂",u"多",u"多少",u"多数",u"多次",u"大力",u"大多数",u"大大",u"大家",u"大批",
            u"大约",u"大量",u"失去",u"她",u"她们",u"她的",u"好的",u"好象",u"如",u"如上所述",u"如下",
            u"如何",u"如其",u"如果",u"如此",u"如若",u"存在",u"宁",u"宁可",u"宁愿",u"宁肯",u"它",
            u"它们",u"它们的",u"它的",u"安全",u"完全",u"完成",u"实现",u"实际",u"宣布",u"容易",u"密切",
            u"对",u"对于",u"对应",u"将",u"少数",u"尔后",u"尚且",u"尤其",u"就",u"就是",u"就是说",
            u"尽",u"尽管",u"属于",u"岂但",u"左右",u"巨大",u"巩固",u"己",u"已经",u"帮助",u"常常",
            u"并",u"并不",u"并不是",u"并且",u"并没有",u"广大",u"广泛",u"应当",u"应用",u"应该",u"开外",
            u"开始",u"开展",u"引起",u"强烈",u"强调",u"归",u"当",u"当前",u"当时",u"当然",u"当着",
            u"形成",u"彻底",u"彼",u"彼此",u"往",u"往往",u"待",u"後来",u"後面",u"得",u"得出",u"得到",
            u"心里",u"必然",u"必要",u"必须",u"怎",u"怎么",u"怎么办",u"怎么样",u"怎样",u"怎麽",u"总之",
            u"总是",u"总的来看",u"总的来说",u"总的说来",u"总结",u"总而言之",u"恰恰相反",u"您",u"意思",
            u"愿意",u"慢说",u"成为",u"我",u"我们",u"我的",u"或",u"或是",u"或者",u"战斗",u"所",
            u"所以",u"所有",u"所谓",u"打",u"扩大",u"把",u"抑或",u"拿",u"按",u"按照",u"换句话说",
            u"换言之",u"据",u"掌握",u"接着",u"接著",u"故",u"故此",u"整个",u"方便",u"方面",u"旁人",
            u"无宁",u"无法",u"无论",u"既",u"既是",u"既然",u"时候",u"明显",u"明确", u"是",u"是否",
            u"是的",u"显然",u"显著",u"普通",u"普遍",u"更加",u"曾经",u"替",u"最后",u"最大",u"最好",
            u"最後",u"最近",u"最高",u"有",u"有些",u"有关",u"有利",u"有力",u"有所",u"有效",u"有时",
            u"有点",u"有的",u"有着",u"有著",u"望",u"朝",u"朝着",u"本",u"本着",u"来",u"来着",u"极了",
            u"构成",u"果然",u"果真",u"某",u"某个",u"某些",u"根据",u"根本",u"欢迎",u"正在",u"正如",
            u"正常",u"此",u"此外",u"此时",u"此间",u"毋宁",u"每",u"每个",u"每天",u"每年",u"每当",
            u"比",u"比如",u"比方",u"比较",u"毫不",u"没有",u"沿",u"沿着",u"注意",u"深入",u"清楚",
            u"满足",u"漫说",u"焉",u"然则",u"然后",u"然後",u"然而",u"照",u"照着",u"特别是",u"特殊",
            u"特点",u"现代",u"现在",u"甚么",u"甚而",u"甚至",u"用",u"由",u"由于",u"由此可见",u"的",
            u"的话",u"目前",u"直到",u"直接",u"相似",u"相信",u"相反",u"相同",u"相对",u"相对而言",u"相应",
            u"相当",u"相等",u"省得",u"看出",u"看到",u"看来",u"看看",u"看见",u"真是",u"真正",u"着",
            u"着呢",u"矣",u"知道",u"确定",u"离",u"积极",u"移动",u"突出",u"突然",u"立即",u"第",u"等",
            u"等等",u"管",u"紧接着",u"纵",u"纵令",u"纵使",u"纵然",u"练习",u"组成",u"经",u"经常",
            u"经过",u"结合",u"结果",u"给",u"绝对",u"继续",u"继而",u"维持",u"综上所述",u"罢了",u"考虑",
            u"者",u"而",u"而且",u"而况",u"而外",u"而已",u"而是",u"而言",u"联系",u"能",u"能否",
            u"能够",u"腾",u"自",u"自个儿",u"自从",u"自各儿",u"自家",u"自己",u"自身",u"至",u"至于",
            u"良好",u"若",u"若是",u"若非",u"范围",u"莫若",u"获得",u"虽",u"虽则",u"虽然",u"虽说",
            u"行为",u"行动",u"表明",u"表示",u"被",u"要",u"要不",u"要不是",u"要不然",u"要么",u"要是",
            u"要求",u"规定",u"觉得",u"认为",u"认真",u"认识",u"让",u"许多",u"论",u"设使",u"设若",
            u"该",u"说明",u"诸位",u"谁",u"谁知",u"赶",u"起",u"起来",u"起见",u"趁",u"趁着",u"越是",
            u"跟",u"转动",u"转变",u"转贴",u"较",u"较之",u"边",u"达到",u"迅速",u"过",u"过去",u"过来",
            u"运用",u"还是",u"还有",u"这",u"这个",u"这么",u"这么些",u"这么样",u"这么点儿",u"这些",
            u"这会儿",u"这儿",u"这就是说",u"这时",u"这样",u"这点",u"这种",u"这边",u"这里",u"这麽",
            u"进入",u"进步",u"进而",u"进行",u"连",u"连同",u"适应",u"适当",u"适用",u"逐步",u"逐渐",
            u"通常",u"通过",u"造成",u"遇到",u"遭到",u"避免",u"那",u"那个",u"那么",u"那么些",u"那么样",
            u"那些",u"那会儿",u"那儿",u"那时",u"那样",u"那边",u"那里",u"那麽",u"部分",u"鄙人",u"采取",
            u"里面",u"重大",u"重新",u"重要",u"鉴于",u"问题",u"防止",u"阿",u"附近",u"限制",u"除",
            u"除了",u"除此之外",u"除非",u"随",u"随着",u"随著",u"集中",u"需要",u"非但",u"非常",u"非徒",
            u"靠",u"顺",u"顺着",u"首先",u"高兴",u"是不是",u"说说",u"，",u"。",u"《",u"》",u"？",u"『",u"！",",",".","!","?",u"、","\"","\"",
            u"月",u"日",u"年","“","…","”",u"】",u"【",u"（",u"）"
        ]
        jieba.load_userdict(WORD_DICT)

    def del_stopwords(self, do=True):
        """
        delete the stopwords
        """
        if do:
            for word in self.cut_text:
                # print word, word in self.stop_words
                if word not in self.stop_words:
                    self.tokens.append(word)
        else:
            for word in self.cut_text:
                self.tokens.append(word)
    
    def is_digit(self, word):
        value = re.compile(r"^[-+]?[0-9]+[\.0-9]*$")
        result = value.match(word)
        if result:
            return False
        else:
            return True

    def del_digit(self):
        self.tokens = filter(self.is_digit, self.tokens)
    
    def is_alpha(self, word):
        result = all(ord(c) < 128 for c in word)
        return not result

    def del_alpha(self):
        self.tokens = filter(self.is_alpha, self.tokens)

    def cut(self, origin_text):
        """
        text : String
        return: generator
        """
        cut_text = jieba.cut(origin_text)
        self.cut_text = cut_text

    def run(self, origin_text):
        """
        origin_text: String
        return: a list of tokens
        """
        self.tokens = []
        self.cut(origin_text)
        self.del_stopwords()
        self.del_digit()
        self.del_alpha()
        return self.tokens


if __name__ == "__main__":
    cut_doc_obj = cutDoc()
    DATA_DIR = "../../data/origin_data"
    DATA_DIR = os.path.abspath(DATA_DIR)
    data_path = os.path.join(DATA_DIR, "all.csv")
    print data_path
    fwrite = open(data_path.replace("all.csv","all_token.csv"), 'w')
    with open(data_path, "r") as fread:
        i = 0
        # while True:
        for line in fread.readlines():
            try:
                line_list = line.strip().split("\t")
                print len(line_list)
                label = line_list[0]
                text = line_list[1]
                # print len(text)
                text_tokens = cut_doc_obj.run(text)
                # print text_tokens
                fwrite.write(' '.join(text_tokens))
                print "processing {0}th line".format(i)
                i+=1
            except BaseException as e:
                msg = traceback.format_exc()
                print msg
                print "=====>Read Done<======"
                break
    fwrite.close()