from pprint import pprint

class Debugger:
    DEBUG_INFO_PATH = r"DebugInfo.txt"
    PPRINT_WIDTH = 120

    debug_file = None

    def init(self):
        self.debug_file = (
            open(self.DEBUG_INFO_PATH, mode="w", encoding="utf-8") )
        
        
        





    def debug_dump(self, heading: str, obj: object):
        if self.debug_file is None:
            return
        print(f"========== {heading} ==========", file=self.debug_file)
        print("", file=self.debug_file)
        if isinstance(obj, str):
            print(obj, file=self.debug_file)
        else:
            pprint(obj, stream=self.debug_file, width=self.PPRINT_WIDTH, sort_dicts=False)
        print("", file=self.debug_file)