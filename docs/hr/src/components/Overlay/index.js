
import React, { useEffect, useState, useRef } from 'react';
import './overlay.css';
import { useLocation } from 'react-router-dom';


const OverLay = () => {

  const [open, setOpen] = useState(false);
  const location = useLocation();
  const modalRef = useRef(null);
  const onCloseModal = () => {
    setOpen(false)
  };


  useEffect(() => {
    const hasModalBeenShown = sessionStorage.getItem('hasModalBeenShown');

    if (!hasModalBeenShown && location.pathname.startsWith('/docs')) {
      setOpen(true);
      sessionStorage.setItem('hasModalBeenShown', 'true');
    }

    const handleClickOutside = (event) => {
      if (modalRef.current && !modalRef.current.contains(event.target)) {
        onCloseModal();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [location.pathname]);

  return (
    open === true && (<div className='modal_over_page' >
      <div id="modal" ref={modalRef}>
        <div className='close_btn_container'>
          <button id="close" onClick={onCloseModal} className="close_btn">
            &#x2715;
          </button>
        </div>
        <div className="modal_card">
          <div>
            <svg xmlns="http://www.w3.org/2000/svg" xmlnsXlink="http://www.w3.org/1999/xlink" width={300} height={60} viewBox="0 0 169 33">
              <defs>
                <clipPath id="logo_svg__c">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__e">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__g">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__i">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__k">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__m">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__o">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__q">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__s">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__u">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__w">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__y">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__A">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__C">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__E">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__G">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__I">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__K">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__M">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__O">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__Q">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__S">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__U">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__W">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__Y">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__aa">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <clipPath id="logo_svg__ac">
                  <path d="M0 0h169v33H0z" />
                </clipPath>
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAjElEQVR4nO3Suw3CQBAFwDXmE7oLUkJKoAAaoAMXRC2kdEBGQg9Eh8VRAqdLfIgZabN92g1eBDSuqw3mdDxFXFZly49rtxlutbegSp62zzyt30WTzuPc//K7FnM/AN8oKc1TUpqnpDRPSWnesjqZd4eI1Bftvvb36jsAAAAAAAAAAAAAAAAAAAAAf+wDL3wvnqQ2544AAAAASUVORK5CYII=" id="logo_svg__b" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAjklEQVR4nO3SoQ0CQRAF0DlCThCKweOxFIXDoKECGqAHKqADHALNciwdLCsuuQ15Lxk1yc8XPwIAAICirvTM6byJ2K6rkvLy3vXDaZRWUCunyz6/+0/Vpdt16r78p9nUBeAXI6V5RkrzjJTmGSnNmxe/z8cuFqtDVdJwfI1RCAAAAAAAAAAAAAAAAABa8gV/ADGPtVxovgAAAABJRU5ErkJggg==" id="logo_svg__d" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAkUlEQVR4nO3YsQkCMRiG4ZzCFWrnAoIrOIOVM7jMOYxg6RZ2Fq6gKwinxglOU/xwKZ4HUia8xVclJQAAAH5qoh/Mz906TU/74gvXxaHZvPvoDhiU++M2v9pP8bl3s7Gbqdtk7AD4x0ipnpFSPSOlekZK9eK/oG5dm1bnZXHA/PKIbgAAAAAAAAAAAAAAAACAKF+qGTJcJp9HBwAAAABJRU5ErkJggg==" id="logo_svg__f" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAiElEQVR4nO3XsQkCQRAF0H+iIKJgC3ZhIzZibi2HXVwJ14JNmBl5cpbgsgYu+F78h5ngJ5MAAAAAAABN62oH5+nYJ7d1Wfhy7VbnoXYX/21ZP3o/JY9NUXTejUmUlCqLXx8AnygpzVNSmqekNO+Lx2n/TLZTWfbwqt8DAAAAAAAAAAAAAAANewPiFQ8ReRn7gAAAAABJRU5ErkJggg==" id="logo_svg__h" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAmUlEQVR4nO3SsQnCQBiG4YtoKguH0IksLBzHBSwEwR1cwkJcwMIpDNhYSM4BjmCakB98HrjmjoOveFMCAAAAAABCq7oe8n1Xp9VlW/5YPqrZ8TroKugjP1+L/Knb8qxPY2/jv0zGHgC/iJTwREp4IiU8kRKeSAlPpIQnUsKbdr40zTvND/vivt3cUjoPuQkAAAAAAAAAAKC/L9JHIbw6ka2VAAAAAElFTkSuQmCC" id="logo_svg__j" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAk0lEQVR4nO3SMQ6BMRzG4Rb5FuEa7uAazmSUGGxcxyqxMZuZJZ+oAzRi6/ePPE/SpR36Dr+UAAAAAACA0HKrj8p9OU+z6ap+uZxzdzu12gFfledmUV7duzr9dT30NmIbDT0AfhEp4YmU8ERKeCIlPJESnkgJT6SEN2n2U79/pPFuW93nw7HZBgAAAAAAAAAA4P98AHCZICEBtEzuAAAAAElFTkSuQmCC" id="logo_svg__l" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAnUlEQVR4nO3SoRHCQBCG0UsGh4Jq6ARLSwjKYAZDBThUSohDgk1gODQihyK34r2ZdSt+8aUEAAAAAACE1tQeMCWP+01qH6vi03DumuX1NtMk+Jaf90t+rcfiDf229k7+r609AH4RKeGJlPBESngiJTyREp5ICU+khCdSwhMp4YmU8ERKeIvaAya9+0Nqd6fiTz52M60BAAAAAABgygfnPS0yNjX/ogAAAABJRU5ErkJggg==" id="logo_svg__n" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAqElEQVR4nO3asQnCUBSG0RujU1gJ1g5j4zRCJsgQlm7hEg7gAIIIgiLGFV7zyE04p77FX3yvexEAAAAAAEBqzdgD5miItolzLIqOr93QdMdf5UmTthx7wCx9N6eI+6Hodv+5RRfbyosmrey1w4hESnoiJT2Rkp5ISU+kpCdS0hMp6YmU9ERKeiIlPZGSnl9QFQzvfhftc1123L+a1eNSeRIAAAAAAEA9f6fCE/WGaQnKAAAAAElFTkSuQmCC" id="logo_svg__p" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAp0lEQVR4nO3asQlCMRSG0eShWFvZOYBjOIMTOIAbCAG3sHMDN3ALa8HGEQTFOMGDFAZvcU59CX/xlUkJAAAAAAAILf97AHHVchjSqrQ1ci01l/2nxw6RMqq+t/eUzou2680jT47LHjuGHo/CL4mU8ERKeCIlPJESnkgJT6SEJ1LCEynhiZTwREp4IiU8H0wYVV+3dcqnWdvx7pmn80vnSQAAAAAAQERfdK8UHEl1GfkAAAAASUVORK5CYII=" id="logo_svg__r" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAn0lEQVR4nO3SsQnCYBCG4ftjwBlcwMINXEPBeYJuZJUlHMAR7AUbQxwhAY05wvPUx/EVbwQAAAAAAJBamXsAy9I3lyp2zWBX5dR1Y3+KlJ/q39tnxGM9fLk/lrq9jvlZfTsKpiZS0hMp6YmU9ERKeiIlPZGSnkhJT6SkJ1LSEynpiZT06rkHsDD9+RDlvhq8e21uEe0fBgEAAAAAAJP7ADVMEGRXS5FoAAAAAElFTkSuQmCC" id="logo_svg__t" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAoElEQVR4nO3ZMQ4BURSG0fsQjV1MoVSKSiKR2YLEMqxKYwGWoJJINBZgDUIx6mkoiPdGzilv9RdfdyMAAAAAAAAAAOCFlHtAlzWPbR0xm7ePq10aHo6ZJv2lQe4B3VYtIk03rVOzvkSI9Jt6uQfAOyKleCKleCKleCKleCKleCKleCKleD5OH2hu+3H0z1XreF+e0mhyzTQJAAAAAADg954mVRLu+vk2dwAAAABJRU5ErkJggg==" id="logo_svg__v" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAmUlEQVR4nO3ZIQ4CMRRF0XYWQMgoHI4toFgO+2EVeCSOsACCR2LHEshAUhSiBmQ/yTlJzVdPXNeUAAAAAAAAAADgi9x6AO2VcbpO+TyvjvfrJk9WQ6NJUCvPciivfqzeY79oveujaz0AfhEp4YmU8ERKeCIlPJESnkgJT6SE58eJVMbtMnWXvjoOu2OenW6NJgEAAAAAAP/kDZfXIb4UYgc+AAAAAElFTkSuQmCC" id="logo_svg__x" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAR0lEQVR4nO3SsQ0AIAwEsQD7zwwjUOYj2RNccVUAAAAAAAAAANDt1lndDcy1uwPgx6TEMynxTEo8kxLPpAAAAAAAAAAAwGwPgpABEI2ucOAAAAAASUVORK5CYII=" id="logo_svg__z" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAoUlEQVR4nO3aOwrCQBSG0RlfnQQ34BrcixtwXa4liNi7DyuxSGUSYi3C2EhywXNgulv8xVdOSgAAAAAAAAAAUJCnHjCW4bk/pFxvy0fNLa/640iT4N3QNpehq9ria0/XqXfyaTb1APhGpIQnUsITKeGJlPBESngiJTyREp5ICU+khCdSwhMp4f3PL6j7vEqL87J41O+6vFk/RpoEAAAAv/ECf+EteP0QRZ0AAAAASUVORK5CYII=" id="logo_svg__B" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAmUlEQVR4nO3YMQ5BURCG0XlCRPPsR2cLtmADSo1d6K1ArET1ohUbkGi06BVu5d0R59RT/MVXTQQAAAAAAAAAAHzQ1B6QzfM6a6M9rYqHj92hGS+6Hib9vWHtAelMttNo5pvy4eUcESLtwaD2ACgRKemJlPRESnoiJT2Rkp5ISU+kpOeZ/+7e3WK0XJcP98fvjwEAAAAAAPgBL1ELECjLZpqWAAAAAElFTkSuQmCC" id="logo_svg__D" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAlklEQVR4nO3aoRHCQBCG0T0miFTA0FcspVAGBUElaQEBJuLoIDcR5BbmPb3iF5/cCAAAAAAAAAAAWFF6D2Cbupyu7avxXo7z4/tr9jH0HsBGZWlHWscaEX8T6aH3AGgRKemJlPRESnoiJT2Rkp5ISU+kpCdS0hMp6YmU9ERKeh5Mfs37co6Y1r/XnrdXxLzPHgAAAPr7AJdRDoBqhLrlAAAAAElFTkSuQmCC" id="logo_svg__F" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAlElEQVR4nO3ZoQ0CMRiG4R6HPwsKAcuQMAHTwA7swhA4JEMg4RKaFI2hghz9xfMkdb/4xOuaEgAAAAAAAAAAfNG1HsA0yrjYpP65rx4+VqduuN7/MAk+lXzcljy8qm/crVtvrZm1HgA1IiU8kRKeSAlPpIQnUsITKeGJlPDmrQcwkXy5pX55qN6NZ79NAAAAAADAb96wwy1IFy4h9wAAAABJRU5ErkJggg==" id="logo_svg__H" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAApUlEQVR4nO3YoQ3CUBSG0fc6AMFgmACFIE1QFSRswiYswRwYFBaHwLMBQZNAIbDBQ9C0r+EcfcUvPnVDAAAAAAAAAACAhNj1gF+976tJiNNh8ugwO8ZF9WxpEg3rf6T1fBfieZk8um3HcVBdW5pEw4quB8A3IiV7IiV7IiV7IiV7IiV7IiV7IiV7/X/mPy5lKDaj5NFpvY/lq25pEgAAAAAAwH/5AIVkFRSMdbGWAAAAAElFTkSuQmCC" id="logo_svg__J" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAmElEQVR4nO3ZoQ3CQBiG4TtIaxmGEbDMRggLsAIzdIMqJFgGICS0ySFwJAVU7xfPk5z7xSdedykBAAAAAAAAAMAXufYA3spzuU5p10xfXB653ffzLYIPZdzeyrgaJt9wONfeWMui9gD4RaSEJ1LCEynhiZTwREp4IiU8kRKeH6cgytBvUj610wfHe26u3YyTAAAAAAAA/vMCuP4pySwoPZEAAAAASUVORK5CYII=" id="logo_svg__L" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAATUlEQVR4nO3SQRHAMAgAQVr8G4lJKiG/wDS7Cu5xEQAAAAAAAAAAAAAAt6iV2d3A/7zdAbBjUsYzKeOZlPFMyl0q8uluAAAAAAAA4LAP4ocCvT7QUE0AAAAASUVORK5CYII=" id="logo_svg__N" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAlklEQVR4nO3SsQkCMRiG4ZwcggtY2bmHEziEq1jYybWCO4g7CSLYWpzgYRwgiF3uR54H0iRFvuJNCQAAAAAAAAAAAAAgsKbWR/lxnKfZritf+nPT3k+1dsBX+ble5mH6Ls6r3469jdgmYw+AX0RKeCIlPJESnkgJT6SEJ1LCEynhtdV+uh2uabFZFffD/lJtAwAAAPyjDziyIXDcNYvZAAAAAElFTkSuQmCC" id="logo_svg__P" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAmElEQVR4nO3SoQ3CQBiG4SshFSgUkyDRrMMQzIAhCBZgA1gA1R1IsAiQ0GOASwOIpn/C8yRn7jefeFMCAAAAAAAAAAAAAAis6jrk63qSZrttcWjnx6o+7PscBV/Jt/s0P+u2fKvN0Nv4L6OhB8AnIiU8kRKeSAlPpIQnUsITKeGJlPDGnZemeaTFaVn8v86XPgcBAAAAP3oDhWQiflfGgvMAAAAASUVORK5CYII=" id="logo_svg__R" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAARklEQVR4nO3SsQ0AIAwEsQD7zwwjUKEPkj3BFVcFAAAAAAAAAAAAAAAAAOTtWiPdwJ9mOgBuTEp7JqU9k9KeSQEAAAB47AD4FwEOwAkqTQAAAABJRU5ErkJggg==" id="logo_svg__T" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAR0lEQVR4nO3SsQ0AIAwEsQD7zwwjUOYj2RNccVUAAAAAAAAAAAAAAAAAQK9bZ3U3MNfuDoAfkxLPpMQzKfFMSjyTAgAAwHQPG+4BEIr5fagAAAAASUVORK5CYII=" id="logo_svg__V" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAWUlEQVR4nO3SywmAMBBAweCn/5IUcrUbC1g7UAiBXXCmgnd4rQEAAAAAAAAAAAAAAAAAc8XRt+wGeBXXumc38B9LdgB8MSnlmZTyTEp5Y5PeZ0zuAAAAAEY9gV8HFTN2C3MAAAAASUVORK5CYII=" id="logo_svg__X" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAATUlEQVR4nO3SQQ0AIAwEwQL+jWASTJTQwIyAyz02AgAAAAAAAAAAAAAAAAAAAEizYrTbH3hLT1+cBzb5mqAoT6SUJ1LKEynliRQAnrcByjgBvCPJPH8AAAAASUVORK5CYII=" id="logo_svg__Z" width={169} height={33} />
                <image xlinkHref="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAhCAYAAACiNouaAAAABmJLR0QA/wD/AP+gvaeTAAAAT0lEQVR4nO3UwQkAIBADQVHsW7A9ezo78CV4yEwFeSwpBQAAAAAAAAAAAAAAAAAAALgixqyvN8BRrNZfb+A/no/0REp6IiU9kZKeSAHgextHnQPir+EocgAAAABJRU5ErkJggg==" id="logo_svg__ab" width={169} height={33} />
                <mask id="logo_svg__ae">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__b" />
                  </g>
                </mask>
                <mask id="logo_svg__ag">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__d" />
                  </g>
                </mask>
                <mask id="logo_svg__ai">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__f" />
                  </g>
                </mask>
                <mask id="logo_svg__ak">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__h" />
                  </g>
                </mask>
                <mask id="logo_svg__am">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__j" />
                  </g>
                </mask>
                <mask id="logo_svg__ao">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__l" />
                  </g>
                </mask>
                <mask id="logo_svg__aq">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__n" />
                  </g>
                </mask>
                <mask id="logo_svg__as">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__p" />
                  </g>
                </mask>
                <mask id="logo_svg__au">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__r" />
                  </g>
                </mask>
                <mask id="logo_svg__aw">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__t" />
                  </g>
                </mask>
                <mask id="logo_svg__ay">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__v" />
                  </g>
                </mask>
                <mask id="logo_svg__aA">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__x" />
                  </g>
                </mask>
                <mask id="logo_svg__aC">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__z" />
                  </g>
                </mask>
                <mask id="logo_svg__aE">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__B" />
                  </g>
                </mask>
                <mask id="logo_svg__aG">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__D" />
                  </g>
                </mask>
                <mask id="logo_svg__aI">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__F" />
                  </g>
                </mask>
                <mask id="logo_svg__aK">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__H" />
                  </g>
                </mask>
                <mask id="logo_svg__aM">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__J" />
                  </g>
                </mask>
                <mask id="logo_svg__aO">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__L" />
                  </g>
                </mask>
                <mask id="logo_svg__aQ">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__N" />
                  </g>
                </mask>
                <mask id="logo_svg__aS">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__P" />
                  </g>
                </mask>
                <mask id="logo_svg__aU">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__R" />
                  </g>
                </mask>
                <mask id="logo_svg__aW">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__T" />
                  </g>
                </mask>
                <mask id="logo_svg__aY">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__V" />
                  </g>
                </mask>
                <mask id="logo_svg__ba">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__X" />
                  </g>
                </mask>
                <mask id="logo_svg__bc">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__Z" />
                  </g>
                </mask>
                <mask id="logo_svg__be">
                  <g filter="url(#logo_svg__a)">
                    <use xlinkHref="#logo_svg__ab" />
                  </g>
                </mask>
                <g id="logo_svg__ad" clipPath="url(#logo_svg__c)">
                  <path d="M19.723 1.605c1.902 1.11 3.804 2.22 5.71 3.317a460.025 460.025 0 0 1-5.695 3.32 460.025 460.025 0 0 1-5.695-3.32c1.906-1.09 3.8-2.195 5.68-3.317Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f4', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__af" clipPath="url(#logo_svg__e)">
                  <path d="M7.457 8.824v-.062a365.663 365.663 0 0 1 5.645-3.27c1.91 1.078 3.8 2.176 5.683 3.301-1.89 1.11-3.785 2.21-5.683 3.305a400.777 400.777 0 0 1-5.645-3.274Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f4', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__ah" clipPath="url(#logo_svg__g)">
                  <path d="M26.316 5.492a.27.27 0 0 1 .18.043 452.851 452.851 0 0 0 5.527 3.211c.032.024.036.047.016.078a475.018 475.018 0 0 1-5.668 3.274c-1.879-1.094-3.758-2.18-5.644-3.274-.04-.02-.04-.039 0-.062 1.867-1.09 3.726-2.184 5.59-3.27Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f5', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aj" clipPath="url(#logo_svg__i)">
                  <path d="M19.695 9.344a255.815 255.815 0 0 1 5.739 3.304 1357.75 1357.75 0 0 0-5.696 3.332c-1.894-1.113-3.789-2.226-5.695-3.332 1.887-1.105 3.773-2.207 5.652-3.304Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f5', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__al" clipPath="url(#logo_svg__k)">
                  <path d="M6.938 9.621a463.903 463.903 0 0 1 5.667 3.305c.02 2.207.02 4.418 0 6.633-1.886-1.102-3.765-2.204-5.652-3.301-.016-2.215-.02-4.426-.016-6.637Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f5', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__an" clipPath="url(#logo_svg__m)">
                  <path d="M32.508 9.621c.039 2.203.043 4.414.015 6.637-1.89 1.097-3.785 2.199-5.675 3.3a312.863 312.863 0 0 1 0-6.632c1.89-1.102 3.777-2.203 5.66-3.305Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7428f5', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__ap" clipPath="url(#logo_svg__o)">
                  <path d="M41.805 9.71c1.703-.132 3.254.286 4.644 1.27.117.106.23.215.332.336a38.638 38.638 0 0 1-1.562 1.637c-.942-.957-2.07-1.308-3.387-1.055-.723.157-1.191.59-1.41 1.297-.078.465.043.86.36 1.196.316.289.675.511 1.077.656.746.2 1.473.43 2.192.703a7.98 7.98 0 0 1 2.504 1.637c.761.847 1.023 1.836.793 2.965-.368 1.742-1.426 2.78-3.18 3.105-2.121.277-4.012-.254-5.676-1.59.516-.535 1.024-1.07 1.535-1.605 1.051.968 2.282 1.332 3.692 1.086a1.912 1.912 0 0 0 1.195-.809c.348-.746.211-1.371-.406-1.883a6.257 6.257 0 0 0-1.977-.797 11.709 11.709 0 0 1-2.55-1.101c-1.829-1.18-2.336-2.797-1.536-4.848.47-.98 1.227-1.64 2.262-1.969.363-.105.73-.183 1.098-.23Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__ar" clipPath="url(#logo_svg__q)">
                  <path d="M85.352 9.988c1.949-.011 3.906 0 5.859.016 2.227.258 3.922 1.34 5.086 3.254.945 1.73 1.137 3.543.578 5.441-.8 2.309-2.379 3.781-4.723 4.422a6.358 6.358 0 0 1-1.093.152c-.887.02-1.782.02-2.672 0 .316-.753.62-1.511.91-2.277.61-.027 1.215-.043 1.824-.047 1.813-.273 3.02-1.273 3.629-3.012.465-1.828.012-3.386-1.352-4.664a4.278 4.278 0 0 0-2.187-.945 185.7 185.7 0 0 0-3.613-.02V23.29h-2.246V9.99Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__at" clipPath="url(#logo_svg__s)">
                  <path d="M135.094 9.988c1.953-.011 3.91 0 5.863.016 2.543.289 4.363 1.601 5.45 3.926.945 2.433.667 4.703-.837 6.82-1.203 1.523-2.773 2.363-4.707 2.523-.89.016-1.777.02-2.672.016.282-.773.575-1.547.88-2.312.605-.008 1.214-.016 1.824-.028 1.968-.316 3.214-1.441 3.718-3.379.27-1.547-.12-2.894-1.168-4.035a4.284 4.284 0 0 0-2.488-1.207 220.28 220.28 0 0 1-3.602-.047c-.015 3.668-.02 7.336-.015 11.008h-2.246V9.99Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__av" clipPath="url(#logo_svg__u)">
                  <path d="M148.031 9.988c2.074-.011 4.153 0 6.227.016 1.293.16 2.258.809 2.898 1.941.73 1.457.656 2.871-.23 4.235.414.28.75.629 1.02 1.054.652 1.133.777 2.32.363 3.578-.618 1.446-1.704 2.266-3.262 2.461-1.317.016-2.633.02-3.945.016.27-.777.562-1.547.875-2.312 1.011-.008 2.023-.016 3.035-.028.812-.199 1.246-.719 1.308-1.562-.043-.79-.437-1.313-1.183-1.555a125.968 125.968 0 0 0-4.86-.047v5.504h-2.246V9.99Zm2.246 2.32c1.297-.003 2.594 0 3.887.02.582.133.973.485 1.172 1.055.203.91-.106 1.578-.926 2-.086.02-.168.043-.246.062-1.293.004-2.586.02-3.871.047-.016-1.058-.02-2.125-.016-3.183Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__ax" clipPath="url(#logo_svg__w)">
                  <path d="M84.195 13.473c.078-.008.137.015.18.078a22.942 22.942 0 0 0 0 2.199c-1.414.129-2.29.887-2.625 2.277-.008 1.743-.02 3.493-.027 5.235-.75.027-1.508.035-2.262.027v-9.543c.746-.004 1.496 0 2.246.02.02.214.04.433.063.652a3.997 3.997 0 0 1 2.425-.945Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__az" clipPath="url(#logo_svg__y)">
                  <path d="M133.941 13.473h.211v2.324c-1.234.062-2.078.664-2.535 1.805-.05.183-.09.363-.12.55-.017 1.715-.02 3.422-.017 5.137h-2.246v-9.543h2.246c-.003.238 0 .473.016.707a4.12 4.12 0 0 1 2.445-.98Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aB" clipPath="url(#logo_svg__A)">
                  <path d="M25.86 13.5c0-.012.011-.023.027-.027.043 2.214.043 4.43 0 6.636.011-2.207 0-4.41-.028-6.609Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#341d59', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aD" clipPath="url(#logo_svg__C)">
                  <path d="M62.91 13.5c1.7-.094 3.067.531 4.113 1.867.965 1.434 1.23 2.996.79 4.68-.504 1.672-1.583 2.777-3.235 3.32a4.761 4.761 0 0 1-3.219-.258.429.429 0 0 1 .043-.277c.25-.594.477-1.195.688-1.805 1.39.453 2.5.075 3.324-1.132.465-.918.465-1.832 0-2.75-.805-1.196-1.89-1.575-3.266-1.153-.96.45-1.515 1.195-1.656 2.25-.015 2.945-.02 5.89-.015 8.836H58.23V13.746h2.247c-.004.246 0 .492.015.734a4.473 4.473 0 0 1 2.418-.98Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aF" clipPath="url(#logo_svg__E)">
                  <path d="M73.203 13.5c1.906-.07 3.399.676 4.477 2.234.738 1.22.965 2.536.683 3.946H71.23c.301.664.797 1.133 1.489 1.394a2.88 2.88 0 0 0 2.23-.199c.778.3 1.555.613 2.324.95-1.214 1.374-2.734 1.921-4.554 1.632-1.922-.445-3.196-1.602-3.813-3.473-.511-2.05-.023-3.808 1.473-5.273a5.127 5.127 0 0 1 2.824-1.211Zm.031 2.328c1.258-.094 2.18.422 2.766 1.547-1.594.012-3.184.02-4.77.012.07-.207.18-.407.317-.582a2.812 2.812 0 0 1 1.687-.977Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aH" clipPath="url(#logo_svg__G)">
                  <path d="M112.684 13.5c2.343-.043 3.949 1.027 4.808 3.215.496 1.57.352 3.078-.422 4.527-1.234 1.88-2.965 2.598-5.175 2.156a3.047 3.047 0 0 1-.82-.324c.265-.687.538-1.37.835-2.047 1.305.434 2.367.098 3.184-1.007a2.85 2.85 0 0 0 .394-1.993c-.254-1.125-.937-1.843-2.047-2.152-1.418-.191-2.425.363-3.023 1.664a3.097 3.097 0 0 0-.168.902c.04 2.868.043 5.743.02 8.61-.754.027-1.512.043-2.266.027V13.746c.75-.004 1.5 0 2.246.02.031.222.043.453.047.687a4.634 4.634 0 0 1 2.387-.953Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aJ" clipPath="url(#logo_svg__I)">
                  <path d="M122.98 13.5c2.102-.05 3.676.836 4.72 2.664a5.403 5.403 0 0 1 .44 3.516c-2.386-.004-4.777 0-7.171.015.691 1.223 1.719 1.703 3.101 1.438.223-.074.438-.16.653-.258.78.3 1.558.613 2.32.95-1.148 1.312-2.598 1.87-4.34 1.663-1.984-.367-3.316-1.504-3.992-3.414-.516-1.855-.16-3.515 1.062-4.984a4.89 4.89 0 0 1 3.207-1.59Zm.028 2.328c1.25-.098 2.16.414 2.73 1.527-1.586.04-3.175.043-4.77.02.434-.852 1.114-1.363 2.04-1.547Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aL" clipPath="url(#logo_svg__K)">
                  <path d="M48.273 13.777h2.274c-.004 1.88 0 3.754.016 5.63.18.796.64 1.366 1.382 1.695 1.196.34 2.086-.032 2.688-1.118a4.99 4.99 0 0 0 .152-.425c.043-1.926.059-3.852.047-5.782h2.246v9.512h-2.246c.004-.234 0-.469-.016-.703-1.195.988-2.527 1.219-3.992.687-1.187-.543-1.965-1.445-2.324-2.707a7.936 7.936 0 0 1-.215-1.253c-.012-1.844-.02-3.692-.012-5.536Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aN" clipPath="url(#logo_svg__M)">
                  <path d="M98.074 13.777h2.246c-.004 1.86 0 3.711.02 5.567.172.91.691 1.515 1.562 1.82 1.192.215 2.051-.23 2.563-1.332a3.32 3.32 0 0 0 .125-.578c.015-1.828.02-3.652.015-5.477h2.247v9.512h-2.247c.004-.234 0-.469-.015-.703-1.223 1.008-2.57 1.227-4.055.66-1.183-.574-1.95-1.5-2.297-2.77a6.026 6.026 0 0 1-.117-.644 195.45 195.45 0 0 1-.047-6.055Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#fefffe', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aP" clipPath="url(#logo_svg__O)">
                  <path d="M19.238 16.777c.04 2.207.04 4.414 0 6.637-.02-.004-.027-.016-.031-.035.031-2.195.043-4.395.031-6.602Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#49218c', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aR" clipPath="url(#logo_svg__Q)">
                  <path d="M26.89 27.234c-.027-2.187-.038-4.379-.027-6.574a533.664 533.664 0 0 1 5.645-3.305 170.86 170.86 0 0 1 0 6.637 630.14 630.14 0 0 1-5.617 3.242Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f6', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aT" clipPath="url(#logo_svg__S)">
                  <path d="M6.938 17.355c1.882 1.086 3.765 2.18 5.648 3.274.043 2.21.043 4.426 0 6.637a726.264 726.264 0 0 0-5.617-3.274c-.032-2.21-.04-4.422-.032-6.637Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f5', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aV" clipPath="url(#logo_svg__U)">
                  <path d="M26.863 20.66c-.011 2.195 0 4.387.028 6.574 0 .016-.008.028-.028.032-.043-2.207-.043-4.41 0-6.606Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#2f1d4c', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aX" clipPath="url(#logo_svg__W)">
                  <path d="M25.86 21.242c0-.02.011-.031.027-.031.043 2.207.043 4.406 0 6.605.011-2.199 0-4.39-.028-6.574Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#341e59', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__aZ" clipPath="url(#logo_svg__Y)">
                  <path d="M13.621 21.242c-.043 2.2-.043 4.403 0 6.606a84.793 84.793 0 0 1-.062-3.317c.011-1.11.02-2.215.027-3.32.02 0 .031.012.035.031Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#452183', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__bb" clipPath="url(#logo_svg__aa)">
                  <path d="M20.242 24.512c-.012 2.199 0 4.39.028 6.578 0 .015-.008.023-.028.027a169.255 169.255 0 0 1 0-6.605Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#4f219a', fillOpacity: 1 }} />
                </g>
                <g id="logo_svg__bd" clipPath="url(#logo_svg__ac)">
                  <path d="M19.238 24.512c.035 2.21.04 4.422.016 6.636a.113.113 0 0 1-.047-.058c.031-2.188.043-4.38.031-6.578Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#49218c', fillOpacity: 1 }} />
                </g>
                <filter id="logo_svg__a" width="100%" height="100%" x="0%" y="0%" filterUnits="objectBoundingBox">
                  <feColorMatrix in="SourceGraphic" values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0" />
                </filter>
              </defs>
              <use xlinkHref="#logo_svg__ad" mask="url(#logo_svg__ae)" />
              <use xlinkHref="#logo_svg__af" mask="url(#logo_svg__ag)" />
              <use xlinkHref="#logo_svg__ah" mask="url(#logo_svg__ai)" />
              <use xlinkHref="#logo_svg__aj" mask="url(#logo_svg__ak)" />
              <use xlinkHref="#logo_svg__al" mask="url(#logo_svg__am)" />
              <use xlinkHref="#logo_svg__an" mask="url(#logo_svg__ao)" />
              <use xlinkHref="#logo_svg__ap" mask="url(#logo_svg__aq)" />
              <use xlinkHref="#logo_svg__ar" mask="url(#logo_svg__as)" />
              <use xlinkHref="#logo_svg__at" mask="url(#logo_svg__au)" />
              <use xlinkHref="#logo_svg__av" mask="url(#logo_svg__aw)" />
              <use xlinkHref="#logo_svg__ax" mask="url(#logo_svg__ay)" />
              <use xlinkHref="#logo_svg__az" mask="url(#logo_svg__aA)" />
              <path d="M25.86 13.5c.027 2.2.038 4.402.027 6.61a1542.58 1542.58 0 0 1-5.66 3.304 313.232 313.232 0 0 1 0-6.637c1.875-1.097 3.753-2.187 5.632-3.277Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f5', fillOpacity: 1 }} />
              <use xlinkHref="#logo_svg__aB" mask="url(#logo_svg__aC)" />
              <use xlinkHref="#logo_svg__aD" mask="url(#logo_svg__aE)" />
              <use xlinkHref="#logo_svg__aF" mask="url(#logo_svg__aG)" />
              <use xlinkHref="#logo_svg__aH" mask="url(#logo_svg__aI)" />
              <use xlinkHref="#logo_svg__aJ" mask="url(#logo_svg__aK)" />
              <path d="M19.238 16.777c.012 2.207 0 4.407-.031 6.602a1395.066 1395.066 0 0 1-5.633-3.27c-.02-2.21-.02-4.425 0-6.636a894.333 894.333 0 0 1 5.664 3.304Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f5', fillOpacity: 1 }} />
              <use xlinkHref="#logo_svg__aL" mask="url(#logo_svg__aM)" />
              <use xlinkHref="#logo_svg__aN" mask="url(#logo_svg__aO)" />
              <use xlinkHref="#logo_svg__aP" mask="url(#logo_svg__aQ)" />
              <use xlinkHref="#logo_svg__aR" mask="url(#logo_svg__aS)" />
              <use xlinkHref="#logo_svg__aT" mask="url(#logo_svg__aU)" />
              <use xlinkHref="#logo_svg__aV" mask="url(#logo_svg__aW)" />
              <use xlinkHref="#logo_svg__aX" mask="url(#logo_svg__aY)" />
              <use xlinkHref="#logo_svg__aZ" mask="url(#logo_svg__ba)" />
              <path d="M13.621 21.242a538.417 538.417 0 0 1 5.617 3.27c.012 2.199 0 4.39-.031 6.578a377.53 377.53 0 0 1-5.586-3.242 169.255 169.255 0 0 1 0-6.606ZM25.86 21.242c.027 2.184.038 4.375.027 6.574a242.093 242.093 0 0 1-5.617 3.274c-.028-2.188-.04-4.38-.028-6.578a241.96 241.96 0 0 1 5.617-3.27Zm0 0" style={{ stroke: 'none', fillRule: 'evenodd', fill: '#7427f6', fillOpacity: 1 }} />
              <use xlinkHref="#logo_svg__bb" mask="url(#logo_svg__bc)" />
              <use xlinkHref="#logo_svg__bd" mask="url(#logo_svg__be)" />
            </svg>
          </div>
          <div className="launch_text">
            <p>📢 On May 1st we will release v0.2 including proper versioning of the docs (the docs are currently outdated). Find all major updates and fixes <a target='_blank' href='https://github.com/SuperDuperDB/superduperdb/blob/main/CHANGELOG.md'>here in the Changelog!</a></p>
          </div>
        </div>
      </div>
    </div >
    ));
};

export default OverLay;