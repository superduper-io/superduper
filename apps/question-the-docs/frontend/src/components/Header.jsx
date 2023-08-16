import superduperdbLogo from '../assets/superduperdb.svg'

const Header = () => {

    return (
        <div className='banner'>
            <a href="https://superduperdb.com" target="_blank">
                <img src={superduperdbLogo} className="logo" alt="SuperDuperDB logo" />
            </a>
            <h1 className='title'>Question the Docs</h1>
        </div>
    )
};

export default Header;