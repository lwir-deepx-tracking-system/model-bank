"""실행 중 여러 부분이 함께 보는 값 보관함 (APP).

한 파일이던 때에는 main() 안의 함수들이 main() 의 지역 변수를 같이 읽고 썼다. 그 함수들을 부분별 파일로 옮기면서,
함께 쓰던 변수는 이 객체의 속성(APP.이름)이 되었다. 값을 넣는 곳과 순서는 main() 그대로다."""


class _Shared:
    """main() 의 지역 변수였던 것 가운데, 다른 파일로 옮긴 함수들이 함께 읽고 쓰는 값이 이 객체의 속성으로 들어간다.

    아직 넣지 않은 이름을 읽으면 NameError 가 난다. 나누기 전에는 main() 안의 변수를 대입 전에 읽으면 NameError 가 났고,
    그 오류를 잡아서 "아직 준비 안 됨"으로 다루는 코드가 있어서 오류 종류를 그대로 맞췄다."""

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        raise NameError("free variable '%s' referenced before assignment in enclosing scope" % name)


APP = _Shared()   # 하나만 만든다. main() 이 시작할 때 비운다.
