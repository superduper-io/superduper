import dataclasses as dc
import typing as t

from superduper import Component


class Streamlit(Component):
    """Streamlit demo function to be deployed on the streamlit server.

    :param demo_func: Callable which builds the demo.
    :param demo_kwargs: key-word arguments to the `demo_func`
    :param default: Set to `True` if this is to be the frontpage.
    :param is_standalone: Set to `True` if this is a standalone page.
    """

    demo_func: t.Callable
    demo_kwargs: t.Dict = dc.field(default_factory=dict)
    default: bool = False
    is_standalone: bool = False

    @property
    def page(self):
        """Get the streamlit page for the multi-page app."""
        import streamlit as st

        def demo_func():
            return self.demo_func(db=self.db, **self.demo_kwargs)

        demo_func.__name__ = self.identifier

        return st.Page(demo_func, title=self.identifier, default=self.default)


if __name__ == '__main__':

    def func_default():
        """Default page."""
        import streamlit as st

        st.title('Welcome to the Superduper Streamlit Server!')

    from superduper import superduper

    db = superduper()

    pages = []
    for demo_name in db.show('streamlit'):
        demo = db.load('streamlit', demo_name)
        demo.init()
        pages.append(demo.page)

    import streamlit as st

    landing = st.Page(func_default, title="About", default=True)
    pg = st.navigation([landing, *pages])
    st.set_page_config(page_title="Superduper demo server", page_icon=":material/edit:")
    pg.run()
