
import AiForm from './AiForm/ui/AiForm';
import { HtmlTagWrapper } from './html-tag-wrapper';


const WrappedWidget = HtmlTagWrapper(AiForm);

const exportedObject = {
    AiForm: WrappedWidget,
};

export default exportedObject;
