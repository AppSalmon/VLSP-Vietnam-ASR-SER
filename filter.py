import re
import os
from itertools import groupby

def number_to_vietnamese(n):
    """
    [Không thay đổi: Hàm chuyển số sang phát âm tiếng Việt, đã thêm ở phiên bản trước.]
    """
    if n == 0:
        return 'không'
    units = ['', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín']
    def read_two_digits(m):
        if m == 0:
            return ''
        tens = ['', 'mười', 'hai mươi', 'ba mươi', 'bốn mươi', 'năm mươi', 'sáu mươi', 'bảy mươi', 'tám mươi', 'chín mươi']
        ten = m // 10
        unit = m % 10
        s = tens[ten]
        if unit == 0:
            return s
        if unit == 1:
            u = 'mốt' if ten > 1 else 'một'
        elif unit == 4:
            u = 'tư' if ten > 1 else 'bốn'
        elif unit == 5:
            u = 'lăm'
        else:
            u = units[unit]
        return s + ' ' + u
    s = ''
    hundred = n // 100
    if hundred > 0:
        s += units[hundred] + ' trăm '
    remain = n % 100
    if remain > 0:
        if hundred > 0 and remain < 10:
            s += 'lẻ '
        s += read_two_digits(remain)
    return s.strip()

def build_mapping_dict():
    """
    [Không thay đổi: Từ điển ánh xạ, với 'covid' map thành 'cô vít' theo yêu cầu trước.]
    """
    mapping_list = {
        'covid': 'cô vít',
        'team': 'thim', 'virtuti': 'vớt tịt', 'czerni': 'chen ni',
        'olympic': 'ô lim píc', 'mc': 'em si', 'ok': 'ô kê', 'drama': 'đờ ram ma',
        'dollar': 'đô la', 'support': 'sụp pót', 'startup': 'sờ tát úp', 'trekking': 'trách kinh',
        'micro': 'mai cờ rô', 'gram': 'gờ ram', 'room': 'rum', 'malay': 'ma lay',
        'fed': 'phét', 'htv': 'hắt tê vê', 'delay': 'đì lây',
        'never': 'ne vờ', 'wave': 'guây', 'livestreamer': 'lai sờ trim mờ', 'fulfilment': 'phun phiu mừng',
        'value': 'vê liu', 'moment': 'mâu mần', 'denish': 'đen nít', 'pakistan': 'pa kít tăng',
        'vaccine': 'vắc xin', 'eu': 'e u', 'schengen': 'tren ghen', 'virus': 'vai rớt',
        'app': 'áp', 'natalie': 'na ta li', 'meow': 'meo', 'cover': 'kó vờ',
        'sale': 'seo', 'traffic': 'trá phịch', 'three': 'thờ ri', 'step': 'sờ tép',
        'formula': 'phó miu lờ', 'draft': 'đờ ráp', 'menu': 'mén niu', 'sars': 'sát',
        'cov': 'si âu vi', 'shark': 'sắc', 'kilo': 'ki lô', 'test': 'tét',
        'angela': 'an gie la', 'gold': 'gôn', 'glucose': 'gờ lu cô dơ', 'romeo': 'rô mê ô',
        'juliet': 'du li ét', 'star': 'sờ ta', 'retzt': 'rét', 'atep': 'a tiếp',
        'israel': 'ích xa ren', 'vitamin': 'vai ta min', 'nato': 'na tô', 'online': 'ón lai',
        'billy': 'biu ly', 'nasa': 'na sa', 'vn': 'vi en', 'top': 'tóp',
        'model': 'mo đồ', 'phone': 'phôn', 'next': 'nếch', 'newton': 'niu tơn',
        'estinfo': 'ét tin pho', 'part': 'pạc', 'time': 'tham', 'yuri': 'gu ri',
        'khonshokov': 'khôn sô cốp', 'usb': 'diu ét bê', 'talk': 'thoóc', 'show': 'sâu',
        'comment': 'cóm men', 'group': 'ghờ rúp', 'spam': 'sờ pam', 'greenpeace': 'gờ rin pi',
        'mom': 'mom', 'australia': 'ốt sờ trây li a', 'imbruglia': 'im bờ ru li a',
        'good': 'gút', 'idea': 'ai đia', 'taxi': 'tắc xi', 'camera': 'ca mê ra',
        'book': 'búc', 'life': 'lai', 'style': 'sờ tai', 'series': 'sía ri',
        'mini': 'mi ni', 'game': 'gêm', 'paralympic': 'pa ra lim píc', 'tokyo': 'tô ky ô',
        'clip': 'cờ líp', 'out': 'ao', 'biden': 'bai đần', 'ray': 'rây',
        'link': 'linh', 'austin': 'ót tin', 'mitsubishi': 'mít su bi si', 'triton': 'trai tờn',
        'raisi': 'rai si', 'krông': 'cờ rông', 'pắc': 'pắc', 'live': 'lai',
        'stream': 'sờ trym', 'guterres': 'gu tê rét', 'ban': 'ban', 'kimoon': 'ki mun',
        'reuters': 'reo tơ', 'brussels': 'bờ rút seo', 'astrazeneca': 'át tra de ni ca',
        'putin': 'pu tin', 'youtube': 'diu túp', 'money': 'mấn nì', 'first': 'phớt',
        'promotion': 'pờ rồ mâu sần', 'mili': 'mi li', 'toilet': 'toi lét', 'make': 'mếch',
        'up': 'ấp', 'paris': 'pa rịt', 'tank': 'tanh', 'tina': 'ti na',
        'grab': 'ghờ ráp', 'singapore': 'sing ga po', 'cambodia': 'cam pu chia', 'mobi': 'mô bi',
        'bio': 'bai ô', 'nano': 'na nô', 'toshiba': 'tô si ba', 'pouyuen': 'pâu duyên',
        'google': 'gu gồ', 'kontum': 'con tum', 'puka': 'pu ca', 'balkan': 'ban căng',
        'ukraine': 'u cờ rai na', 'william': 'quiu li am', 'shakespeare': 'séc bia', 'block': 'bờ lóc',
        'delta': 'đeon ta', 'lgbt': 'eo gi bi ti', 'karaoke': 'ka ra ô kê', 'cafe': 'cà phê',
        'vnc': 'vi en si', 'feel': 'phiu', 'champion': 'cham pi ần', 'antonio': 'an tô ni ô',
        'new': 'niu', 'york': 'doóc', 'alo': 'a lô', 'taj': 'tách',
        'mahal': 'ma han', 'bob': 'bốp', 'dylan': 'đai lần', 'shopping': 'sốp ping',
        'gucci': 'gu chì', 'supply': 'sập lai', 'lou': 'lu', 'iraq': 'i rắc',
        'sergei': 'séc gây', 'shoigu': 'sô gu', 'ocean': 'âu sần', 'park': 'pắc',
        'tiktok': 'tích tốc'
    }
    return {k.lower(): v for k, v in mapping_list.items()}

MAPPING_DICT = build_mapping_dict()

def normalize_text(text: str) -> str:
    """
    Hàm chuẩn hóa văn bản.
    [Không thay đổi: Giữ nguyên sub cho covid-?19, map words, Bước 3 từ đơn.]
    [Chỉnh sửa: Redesign Bước 4 để từ cụm ngắn (2) đến dài (n//2), và trong mỗi length, while changed để repeat scan đến khi không còn xóa (count>2). 
    Update words sau mỗi scan, đảm bảo brute force chính xác cho lặp chồng chéo, chỉ xóa liên tiếp >2. Quan trọng để xử lý đa tầng mà không miss.]
    """
    # Bước 0: Tiền xử lý các biến thể của 'covid'
    text = re.sub(r'\bcovid-?19\b', 'cô vít mười chín', text, flags=re.IGNORECASE)
    # Bước 1: Chuyển về chữ thường và loại bỏ ký tự lặp
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Bước 2: Tách từ và ánh xạ theo từ điển + xử lý số
    original_words = text.split()
    mapped_words = []
    for word in original_words:
        if word.isdigit():
            num = int(word)
            viet = number_to_vietnamese(num)
            mapped_words.extend(viet.split())
        else:
            mapped = MAPPING_DICT.get(word, word)
            mapped_words.extend(mapped.split())

            
    # Bước 3: Xử lý lặp từ đơn
    single_word_cleaned = []
    # print("mapped_words", mapped_words)
    for _, group in groupby(mapped_words):
        word_group = list(group)
        # print(word_group)
        if len(word_group) > 2:
            # print("->", word_group[0])
            single_word_cleaned.append(word_group[0])
        else:
            single_word_cleaned.extend(word_group)
    # print("B1: ", single_word_cleaned)
    # Bước 4: Xử lý lặp cụm từ và cụm con - Sửa theo yêu cầu mới: từ ngắn đến dài, repeat đến fix point cho mỗi length
    words = single_word_cleaned
    n = len(words)
    repeated_phrases = []  # Lưu trữ các cụm đã được xác định là lặp
    for length in range(2, n//2 + 1):  # Từ ngắn đến dài
        changed = True
        while changed:
            changed = False
            final_words = []
            i = 0
            while i < n:
                if i + length > n:
                    final_words.append(words[i])
                    i += 1
                    continue
                phrase = words[i : i + length]
                if i + 2 * length <= n and phrase == words[i + length : i + 2 * length]:
                    # Tính tổng số lần lặp liên tiếp
                    count = 1  # Bao gồm cụm đầu tiên
                    current_pos = i + length
                    while current_pos + length <= n and words[current_pos : current_pos + length] == phrase:
                        count += 1
                        current_pos += length
                    # Quyết định giữ bao nhiêu dựa trên count
                    if count == 2:
                        final_words.extend(phrase)  # Giữ cụm 1
                        final_words.extend(phrase)  # Giữ cụm 2
                    else:  # count > 2
                        final_words.extend(phrase)  # Chỉ giữ 1
                        changed = True  # Có xóa, cần repeat
                    if count > 1:
                        repeated_phrases.append(phrase)  # Lưu để check cụm con
                    i = current_pos  # Skip qua tất cả lặp
                else:
                    final_words.append(words[i])
                    i += 1
            words = final_words  # Update words cho iteration sau
            n = len(words)  # Update n
    # print("B2: ", words)  # Giờ final_words là words sau tất cả
    # Bước 5: Kiểm tra cụm con ở cuối chuỗi
    result_words = []
    i = 0
    while i < len(words):
        is_subphrase = False
        current_phrase = words[i:]
        for phrase in repeated_phrases:
            if len(current_phrase) < len(phrase) and current_phrase == phrase[:len(current_phrase)]:
                is_subphrase = True
                i = len(words)  # Bỏ qua phần còn lại
                break
        if not is_subphrase:
            result_words.append(words[i])
            i += 1
    return ' '.join(result_words).strip()

def process_tsv_file(input_path: str, output_path: str):
    """
    Hàm xử lý file, không thay đổi.
    """
    print(f"Bắt đầu xử lý file: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for i, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    filename, sentiment, original_text = parts
                    normalized_transcript = normalize_text(original_text)
                    outfile.write(f"{filename}\t{sentiment}\t{normalized_transcript}\n")
                else:
                    print(f"Cảnh báo: Dòng {i+1} có định dạng không hợp lệ và đã được bỏ qua: '{line}'")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file đầu vào tại '{input_path}'.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
        return
    print("-" * 50)
    print(f"✅ Xử lý hoàn tất!")
    print(f"File đã chuẩn hóa được lưu tại: {output_path}")
    print("-" * 50)

# --- Điểm bắt đầu của script ---
if __name__ == "__main__":
    input_file = "results_private_pro.tsv"
    output_file = "results_private_pro_clean_not_cut.tsv"
   
    process_tsv_file(input_path=input_file, output_path=output_file)
    
    # original_text = "mày làm giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em giải quyết mấy em"
    # original_text = "ở chính nợ ở quần chính luôn quần chính luôn một người phu nhượng người ở quần chính"
    # normalized_transcript = normalize_text(original_text)
    # print(normalized_transcript)