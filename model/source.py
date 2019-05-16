from pathlib import Path
import xml.etree.ElementTree as ET


class SourceContext:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.filepath = None

    @staticmethod
    def get_some_context(src, start, end):
        context_extractor = LTFSourceContext(src)
        if context_extractor.doc_exists():
            return context_extractor.query_context(start, end)
        context_extractor = TextSourceContext(src)
        if context_extractor.doc_exists():
            return context_extractor.query_context(start, end)
        return ''

    def doc_exists(self):
        return self.filepath and self.filepath.is_file()

    def query_context(self, start, end):
        raise NotImplementedError


class LTFSourceContext(SourceContext):
    source_path = Path('/lfs1/gaia/m9copora/ltf')

    def __init__(self, doc_id):
        super().__init__(doc_id)
        self.filepath = self.source_path / (doc_id + '.ltf.xml')

    def query_context(self, start, end):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        texts = []
        for child in root.findall('./DOC/TEXT/SEG'):
            seg_start, seg_end = int(child.get('start_char')), int(child.get('end_char'))
            if seg_end < start:
                continue
            if seg_start > end:
                break
            text = child.find('ORIGINAL_TEXT').text
            texts.append(text)
        return ' '.join(texts)

    def query(self, start, end):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        for child in root.findall('./DOC/TEXT/SEG'):
            seg_start, seg_end = int(child.get('start_char')), int(child.get('end_char'))
            if seg_end < start:
                continue
            if seg_start > end:
                break
            for c in child.findall('TOKEN'):
                s = int(c.get('start_char'))
                e = int(c.get('end_char'))
                if s == start and e == end:
                    text = c.text
                    break
            if text:
                break
        return text


class TextSourceContext(SourceContext):
    source_path = Path('rsd')

    def __init__(self, doc_id):
        super().__init__(doc_id)
        self.filepath = self.source_path / (doc_id + '.rsd.txt')

    def query_context(self, start, end, length=160):
        """
        Get context, front<--------><em>start-end</em><------->back
        """
        end = end+1
        with open(self.filepath) as f:
            data = f.read()
            front, back = self.calculate_double_side_length(start, end, length, len(data))
            snippet = data[front:start] + '<em>' + data[start:end] + '</em>' + data[end:back]
            if front != 0: snippet = '......' + snippet
            if back != len(data): snippet += '......'
            snippet = snippet.replace('\n', ' ')
            return snippet

    @staticmethod
    def calculate_double_side_length(start, end, length, total):
        """
        Calculate the length of look ahead and look back.
        """
        interval = end - start + 1
        if interval > length:
            return 0, 0
        forward = backward = (length - interval) //2
        if start < forward:
            backward += forward - start
            return 0, min(total, end + backward)
        if total < end + backward:
            forward += backward + end - total
            return max(0, start - forward), total
        return start - forward, end + backward


if __name__ == '__main__':
    sc = TextSourceContext('HC000ZXSM')
    print(sc.doc_exists())
    print(sc.query_context(1075, 1083))
