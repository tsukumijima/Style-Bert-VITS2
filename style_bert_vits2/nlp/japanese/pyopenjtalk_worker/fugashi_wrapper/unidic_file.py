# dicrc(mecabの辞書のリソースファイル)
DICRC = r"""
;    Copyright (c) 2011-2017, The UniDic Consortium
;    All rights reserved.
;
;    Redistribution and use in source and binary forms, with or without
;    modification, are permitted provided that the following conditions are
;    met:
;
;    * Redistributions of source code must retain the above copyright
;    notice, this list of conditions and the following disclaimer.
;
;    * Redistributions in binary form must reproduce the above copyright
;    notice, this list of conditions and the following disclaimer in the
;    documentation and/or other materials provided with the
;    distribution.
;
;    * Neither the name of the UniDic Consortium nor the names of its
;    contributors may be used to endorse or promote products derived
;    from this software without specific prior written permission.
;
;    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
;    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
;    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
;    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
;    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
;    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
;    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
;    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
;    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
;    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
;    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
; List of features
; f[0]:  pos1
; f[1]:  pos2
; f[2]:  pos3
; f[3]:  pos4
; f[4]:  cType
; f[5]:  cForm
; f[6]:  lForm
; f[7]:  lemma
; f[8]:  orth
; f[9]:  pron
; f[10]: orthBase
; f[11]: pronBase
; f[12]: goshu
; f[13]: iType
; f[14]: iForm
; f[15]: fType
; f[16]: fForm
; f[17]: iConType
; f[18]: fConType
; f[19]: type
; f[20]: kana
; f[21]: kanaBase
; f[22]: form
; f[23]: formBase
; f[24]: aType
; f[25]: aConType
; f[26]: aModType
; f[27]: lid
; f[28]: lemma_id

cost-factor = 700
bos-feature = BOS/EOS,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*
eval-size = 10
unk-eval-size = 4

config-charset = utf8
dictionary-charset = utf8

; userdic = {user_dict_path}

output-format-type = unidic22

node-format-unidic22 = %m\t%f[0],%f[1],%f[2],%f[3],%f[4],%f[5],%f[6],%f[7],%f[8],%f[9],%f[10],%f[11],%f[12],"%f[13]","%f[14]","%f[15]","%f[16]","%f[17]","%f[18]",%f[19],%f[20],%f[21],%f[22],%f[23],"%f[24]","%f[25]","%f[26]",%f[27],%f[28]\n
unk-format-unidic22 = %m\t%f[0],%f[1],%f[2],%f[3],%f[4],%f[5]\n
bos-format-unidic22 =
eos-format-unidic22 = EOS\n

node-format-verbose = surface:%m\tpos1:%f[0]\tpos2:%f[1]\tpos3:%f[2]\tpos4:%f[3]\tcType:%f[4]\tcForm:%f[5]\tlForm:%f[6]\tlemma:%f[7]\torth:%f[8]\tpron:%f[9]\torthBase:%f[10]\tpronBase:%f[11]\tgoshu:%f[12]\tiType:%f[13]\tiForm:%f[14]\tfType:%f[15]\tfForm:%f[16]\tiConType:%f[17]\tfConType:%f[18]\tlType:%f[19]\tkana:%f[20]\tkanaBase:%f[21]\tform:%f[22]\tformBase:%f[23]\taType:%f[24]\taConType:%f[25]\taModType:%f[26]\tlid:%f[27]\tlemma_id:%f[28]\n
unk-format-verbose = surface:%m\tpos1:%f[0]\tpos2:%f[1]\tpos3:%f[2]\tpos4:%f[3]\tcType:%f[4]\tcForm:%f[5]\n
bos-format-verbose =
eos-format-verbose = EOS\n

node-format-chamame = \t%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[23]\t%f[12]\n\n
unk-format-chamame = \t%m\t\t\t%m\t未知語\t\t\t\t\n
bos-format-chamame = B
eos-format-chamame =
"""


# node:
# $1: pos1
# $2: pos2
# $3: pos3
# $4: pos4
# $5: cType
# $6: cForm
# $7: lForm
# $8: lemma
# $9: orth
# $10: pron
# $11: orthBase
# $12: pronBase
# $13: goshu
# $14: iType
# $15: iForm
# $16: fType
# $17: fForm
# $18: iConType
# $19: fConType
# $20: type
# $21: kana
# $22: kanaBase
# $23: form
# $24: formBase
# $25: aType
# $26: aConType
# $27: aModType
# unk:
# $1: pos1
# $2: pos2
# $3: pos3
# $4: pos4
# $5: cType
# $6: cForm

RWRITE_DEF = r"""[unigram rewrite]
BOS/EOS,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,*,*,*,*,*,*,BOS/EOS,BOS/EOS,BOS/EOS,*,*,BOS/EOS,*,*,*
*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$7,$8,$9,$11,$10,$12,$13,$25,$26,$27
*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,*,*,*,*,*,*,*

[left rewrite]
BOS/EOS,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,*,*,*,*,*,BOS/EOS,BOS/EOS,BOS/EOS,*,*,*,*,*,*
助詞,*,*,*,*,*,*,(の|に|を|て|は|と|が|で|も|の|から|か|が|ね|か|けれど|など|って|と|ば|や|まで|へ|から|より|だけ|な|たり|よ|くらい|ながら|し|ほど|しか),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
助動詞,*,*,*,*,*,*,(だ|た|ます|です|れる|ず|ない|てる|られる|べし|たい|り|せる|ちゃう),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
名詞,助動詞語幹,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
形状詞,助動詞語幹,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
動詞,非自立可能,*,*,*,*,*,(為る|居る|有る|成る|見る|行く|来る|出来る|得る|遣る|仕舞う|呉れる|出す|置く|致す|付く|頂く|付ける|貰う|掛ける|続く|始める|続ける|御座る|終わる),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
動詞,一般,*,*,*,*,*,(於く),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
形容詞,非自立可能,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
接尾辞,*,*,*,*,*,*,(的|年|者|月|さん|日|パーセント|人|つ|等|日|円|等|化|達|人|さ|性|回|時|氏|所|生|方|分|長|党|目|中|省|歳|内|年度|国|家|後|部|上|車|権|度|力|員|費|書|用|物|型|業|間|メートル|庁|箇月|番|局|機|年間|館|件|時間|社),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
接頭辞,*,*,*,*,*,*,(第|御|約|不|大|新|各|小|御|非),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
名詞,数詞,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
補助記号,*,*,*,*,*,*,*,．,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,*,*,$13,$16,$17,$18,$25,$26,$27
*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,*,*,*,*,*,*

[right rewrite]
BOS/EOS,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,*,*,*,*,*,BOS/EOS,BOS/EOS,BOS/EOS,*,*,*,*,*,*
助詞,*,*,*,*,*,*,(の|に|を|て|は|と|が|で|も|の|から|か|が|ね|か|けれど|など|って|と|ば|や|まで|へ|から|より|だけ|な|たり|よ|くらい|ながら|し|ほど|しか),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
助動詞,*,*,*,*,*,*,(だ|た|ます|です|れる|ず|ない|てる|られる|べし|たい|り|せる|ちゃう),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
名詞,助動詞語幹,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
形状詞,助動詞語幹,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
動詞,非自立可能,*,*,*,*,*,(為る|居る|有る|成る|見る|行く|来る|出来る|得る|遣る|仕舞う|呉れる|出す|置く|致す|付く|頂く|付ける|貰う|掛ける|続く|始める|続ける|御座る|終わる),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
動詞,一般,*,*,*,*,*,(於く),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$16,$17,$18,$25,$26,$27
形容詞,非自立可能,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
接尾辞,*,*,*,*,*,*,(的|年|者|月|さん|日|パーセント|人|つ|等|日|円|等|化|達|人|さ|性|回|時|氏|所|生|方|分|長|党|目|中|省|歳|内|年度|国|家|後|部|上|車|権|度|力|員|費|書|用|物|型|業|間|メートル|庁|箇月|番|局|機|年間|館|件|時間|社),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
接頭辞,*,*,*,*,*,*,(第|御|約|不|大|新|各|小|御|非),*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
名詞,数詞,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
補助記号,*,*,*,*,*,*,*,．,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,$9,$10,$13,$14,$15,$19,$25,$26,$27
*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,*,*,$13,$14,$15,$19,$25,$26,$27
*,*,*,*,*,*	$1,$2,$3,$4,$5,$6,*,*,*,*,*,*"""
